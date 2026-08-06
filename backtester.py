import pandas as pd
import numpy as np
from indicators import engineer_features
from risk_manager import RiskManager

class Backtester:
    def __init__(self, risk_manager=None):
        self.rm = risk_manager or RiskManager()

    @staticmethod
    def calculate_metrics(returns_series, equity_curve, trades, initial_capital, num_days):
        returns = returns_series.dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0.0
        downside = returns[returns < 0]
        sortino = np.sqrt(252) * returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0.0
        
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = abs(float(drawdown.min()))
        
        cagr = (equity_curve.iloc[-1] / initial_capital) ** (252 / num_days) - 1 if num_days > 0 and initial_capital > 0 else 0.0
        
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(wins) / len(trades)
            gross_profit, gross_loss = sum(t['pnl'] for t in wins), abs(sum(t['pnl'] for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = profit_factor = 0.0
            
        volatility = returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
        
        return {
            'sharpe': round(sharpe, 4), 'sortino': round(sortino, 4), 'max_drawdown': round(max_dd * 100, 2),
            'cagr': round(cagr * 100, 2), 'win_rate': round(win_rate * 100, 2), 'profit_factor': round(profit_factor, 4),
            'volatility': round(volatility * 100, 2), 'total_return': round(total_return, 2), 'total_trades': len(trades),
            'initial_capital': initial_capital, 'final_capital': round(float(equity_curve.iloc[-1]), 2),
        }

    def run(self, df_1d, kod, initial_capital=10000, strategy="ml", ml_predictor=None):
        df = df_1d.copy()
        if len(df) < 100: return {"error": f"Yetersiz veri: {len(df)} satır"}
        feat = engineer_features(df)
        
        cash = float(initial_capital)
        peak_equity = float(initial_capital)
        position, entry_price, entry_cost, stop_loss, take_profit = 0, 0.0, 0.0, 0.0, 0.0
        highest_price, trailing_stop = 0.0, 0.0
        entry_date = None
        trades, equity_dates, equity_values = [], [], []

        start_idx = 50
        ml_start = max(start_idx, ml_predictor.split_idx) if strategy == "ml" and ml_predictor and ml_predictor.is_trained else start_idx

        for i in range(ml_start, len(feat) - 1):
            row, next_row = feat.iloc[i], feat.iloc[i + 1]
            date_str = str(row.name) if hasattr(row, 'name') else f"Day_{i}"
            price, atr = float(row['Close']), float(row['ATR']) if row['ATR'] > 0 else float(row['Close']) * 0.02

            current_equity = cash + (position * price if position > 0 else 0)
            peak_equity = max(peak_equity, current_equity)

            if not self.rm.check_max_drawdown(current_equity, peak_equity):
                if position > 0:
                    exit_cost = self.rm.calculate_costs(position, price)
                    proceeds = position * price - exit_cost
                    pnl = proceeds - (position * entry_price + entry_cost)
                    trades.append({'kod': kod, 'entry_date': str(entry_date), 'entry_price': round(entry_price, 2), 'exit_date': date_str, 'exit_price': round(price, 2), 'shares': position, 'pnl': round(pnl, 2), 'return_pct': round((pnl / (position * entry_price + entry_cost)) * 100, 2), 'side': 'LONG', 'reason': 'Max drawdown stop'})
                    cash += proceeds
                    position = 0
                equity_dates.append(date_str)
                equity_values.append(cash)
                continue

            signal = 0
            if strategy == "ml" and ml_predictor and ml_predictor.is_trained:
                feat_row = feat[ml_predictor.FEATURE_COLS].iloc[i:i+1]
                pred, proba = int(ml_predictor.model.predict(feat_row)[0]), float(ml_predictor.model.predict_proba(feat_row)[0].max())
                if pred == 1 and proba > 0.55: signal = 1
            else:
                rsi_val, bb_pctb = float(row['RSI']), float(row['BB_PctB'])
                if rsi_val < 30 and bb_pctb < 0.1: signal = 1
                elif rsi_val > 70 and bb_pctb > 0.9: signal = -1

            next_price = float(next_row['Close'])
            if position > 0:
                if next_price > highest_price:
                    highest_price = next_price
                    new_trailing = highest_price - (atr * self.rm.stop_loss_atr_mult)
                    if new_trailing > trailing_stop: trailing_stop = new_trailing

                exit_reason = None
                if next_price <= trailing_stop and trailing_stop > 0: exit_reason = 'Trailing stop'
                elif next_price <= stop_loss: exit_reason = 'Stop-loss'
                elif next_price >= take_profit: exit_reason = 'Take-profit'
                elif signal == -1: exit_reason = 'Signal sell'

                if exit_reason:
                    exit_cost = self.rm.calculate_costs(position, next_price)
                    proceeds = position * next_price - exit_cost
                    pnl = proceeds - (position * entry_price + entry_cost)
                    trades.append({'kod': kod, 'entry_date': str(entry_date), 'entry_price': round(entry_price, 2), 'exit_date': str(next_row.name) if hasattr(next_row, 'name') else f"Day_{i+1}", 'exit_price': round(next_price, 2), 'shares': position, 'pnl': round(pnl, 2), 'return_pct': round((pnl / (position * entry_price + entry_cost)) * 100, 2), 'side': 'LONG', 'reason': exit_reason})
                    cash += proceeds
                    position = 0

            if signal == 1 and position == 0:
                shares, sl, tp = self.rm.calculate_position_size(cash, next_price, atr)
                if shares > 0:
                    buy_cost = self.rm.calculate_costs(shares, next_price)
                    if shares * next_price + buy_cost <= cash:
                        cash -= (shares * next_price + buy_cost)
                        position, entry_price, entry_cost = shares, next_price, buy_cost
                        stop_loss, take_profit, highest_price, trailing_stop = sl, tp, next_price, sl
                        entry_date = str(next_row.name) if hasattr(next_row, 'name') else f"Day_{i+1}"

            equity = cash + (position * next_price if position > 0 else 0)
            equity_dates.append(date_str)
            equity_values.append(equity)
            peak_equity = max(peak_equity, equity)

        if position > 0:
            last_price = float(feat.iloc[-1]['Close'])
            exit_cost = self.rm.calculate_costs(position, last_price)
            proceeds = position * last_price - exit_cost
            pnl = proceeds - (position * entry_price + entry_cost)
            trades.append({'kod': kod, 'entry_date': str(entry_date), 'entry_price': round(entry_price, 2), 'exit_date': str(feat.index[-1]), 'exit_price': round(last_price, 2), 'shares': position, 'pnl': round(pnl, 2), 'return_pct': round((pnl / (position * entry_price + entry_cost)) * 100, 2), 'side': 'LONG', 'reason': 'Backtest end'})
            cash += proceeds
            if equity_values: equity_values[-1] = cash

        equity_curve = pd.Series(equity_values, dtype=float)
        returns_series = equity_curve.pct_change().dropna()
        metrics = self.calculate_metrics(returns_series, equity_curve, trades, initial_capital, len(equity_values))
        metrics['kod'] = kod
        metrics['equity_curve'] = list(zip(equity_dates, [round(float(v), 2) for v in equity_values]))

        return {'metrics': metrics, 'trades': trades, 'equity_curve': equity_curve, 'equity_dates': equity_dates}
