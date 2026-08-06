class RiskManager:
    def __init__(self, risk_per_trade=0.02, max_position_pct=0.25, stop_loss_atr_mult=2.0, take_profit_atr_mult=3.0, max_daily_loss_pct=0.05, commission_pct=0.001, slippage_pct=0.0005):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_daily_loss_pct = max_daily_loss_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def calculate_position_size(self, portfolio_value, entry_price, atr):
        risk_amount = portfolio_value * self.risk_per_trade
        stop_distance = atr * self.stop_loss_atr_mult
        if stop_distance <= 0 or entry_price <= 0:
            return 0, 0.0, 0.0

        max_shares_by_risk = int(risk_amount / stop_distance)
        max_shares_by_capital = int((portfolio_value * self.max_position_pct) / entry_price)
        shares = min(max_shares_by_risk, max_shares_by_capital)
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (atr * self.take_profit_atr_mult)
        return shares, round(stop_loss, 2), round(take_profit, 2)

    def calculate_costs(self, shares, price):
        return shares * price * (self.commission_pct + self.slippage_pct)

    def check_max_drawdown(self, portfolio_value, peak_value):
        if peak_value <= 0: return True
        drawdown = (peak_value - portfolio_value) / peak_value
        return drawdown < self.max_daily_loss_pct
