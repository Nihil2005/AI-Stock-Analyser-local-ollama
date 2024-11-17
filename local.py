import yfinance as yf
from langchain_community.llms import Ollama
from typing import Dict, List
from datetime import datetime


class WealthOllamaAdvisor:
    def __init__(self):
        # Initialize Ollama model
        self.llm = Ollama(model="stablelm-zephyr")

    def get_market_insights(self, symbol: str) -> Dict:
        """Get AI-powered market insights using Ollama for Indian stocks"""
        nse_symbol = f"{symbol}.NS"
        stock = yf.Ticker(nse_symbol)

        try:
            # Fetch real-time stock data
            hist = stock.history(period="6mo")
            current_price = stock.info.get("currentPrice", None)
            market_cap = stock.info.get("marketCap", None)

            if hist.empty or not current_price:
                return {"error": f"No data found for symbol {symbol}"}

        except Exception as e:
            return {"error": f"Error fetching stock data for {symbol}: {e}"}

        # Add Indian market-specific context with current data
        market_context = f"""
        Stock: {symbol} (NSE)
        Current Price: ₹{current_price:.2f}
        6-Month High: ₹{hist['High'].max():.2f}
        6-Month Low: ₹{hist['Low'].min():.2f}
        Market Cap: ₹{market_cap:,} (if available)
        Volume (latest): {hist['Volume'].iloc[-1]:,}
        """

        # AI prompt for market insights
        prompt = f"""
        As a financial expert familiar with the Indian stock market, analyze the following market data and provide:
        1. Current market sentiment in India.
        2. Key risks specific to {symbol} and the Indian market.
        3. Growth potential and sectoral trends.
        4. Investment recommendation (Buy, Hold, or Sell).

        Data:
        {market_context}
        """

        response = self.llm.invoke(prompt)
        return {"insights": response}

    def create_wealth_strategy(self, user_data: Dict) -> Dict:
        """Generate personalized wealth-building strategy for Indian investors"""
        user_context = f"""
        Profile:
        - Age: {user_data['age']}
        - Income: ₹{user_data['income']:,}
        - Risk Tolerance: {user_data['risk_tolerance']}/10
        - Investment Goals: {user_data['goals']}
        - Time Horizon: {user_data['time_horizon']} years
        """

        prompt = f"""
        Create a wealth-building strategy for this Indian investor:
        {user_context}
        
        Include:
        1. Suggested asset allocation (Indian equity, debt, and gold).
        2. Specific investment vehicles (e.g., mutual funds, stocks, FDs, PPF, NPS).
        3. Tax optimization strategies (under Indian tax laws).
        4. Risk management suggestions.
        5. Specific steps aligned with current market conditions.
        """

        response = self.llm.invoke(prompt)
        return {"strategy": response}

    def get_ai_predictions(self, symbols: List[str]) -> Dict:
        """Get AI predictions for multiple Indian stocks"""
        predictions = {}
        for symbol in symbols:
            nse_symbol = f"{symbol}.NS"
            stock = yf.Ticker(nse_symbol)

            try:
                data = stock.history(period="6mo")
                current_price = stock.info.get("currentPrice", None)

                if data.empty or not current_price:
                    predictions[symbol] = "No sufficient data available"
                    continue

            except Exception as e:
                predictions[symbol] = f"Error fetching data for {symbol}: {e}"
                continue

            prompt = f"""
            Analyze the following Indian stock's recent performance and provide a 6-month prediction:
            Symbol: {symbol} (NSE)
            Current Price: ₹{current_price:.2f}
            6-Month High: ₹{data['High'].max():.2f}
            6-Month Low: ₹{data['Low'].min():.2f}
            
            Consider current market conditions, economic trends, and sectoral developments.
            """

            response = self.llm.invoke(prompt)
            predictions[symbol] = response

        return predictions


def main():
    advisor = WealthOllamaAdvisor()

    # Example user profile
    user_profile = {
        "age": 35,
        "income": 1800000,  # In INR
        "risk_tolerance": 8,
        "goals": "Build long-term wealth for retirement",
        "time_horizon": 20
    }

    # Generate a wealth-building strategy
    print("\nFetching wealth strategy...")
    strategy = advisor.create_wealth_strategy(user_profile)
    print("\nStrategy:", strategy["strategy"])

    # Get current market insights for Reliance Industries
    print("\nFetching market insights for Reliance...")
    insights = advisor.get_market_insights("RELIANCE")
    print("\nInsights:", insights["insights"])

    # Get AI predictions for multiple stocks
    print("\nFetching predictions for multiple stocks...")
    predictions = advisor.get_ai_predictions(["RELIANCE", "TCS", "INFY"])
    for symbol, prediction in predictions.items():
        print(f"\nPrediction for {symbol}: {prediction}")


if __name__ == "__main__":
    main()
