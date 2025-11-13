import pandas as pd
import xgboost as xgb
from backtesting import Backtest, Strategy

# This strategy will integrate our XGBoost model to make trading decisions.
# A key challenge is making our feature data available within the strategy's scope.
class MLStrategy(Strategy):
    def init(self):
        """
        Initialize the strategy.
        """
        # Load the pre-trained XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model('src/tier1/modeling/xgboost_baseline_v1.json')

        # Prepare the feature set (X) that the model will use for predictions.
        # We must explicitly exclude the OHLCV columns required by the backtesting library,
        # as the model was not trained on them.
        ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
        features = [c for c in self.data.df.columns if c not in ohlcv and not c.startswith('btc_') and c != 'label']
        self.features = self.data.df[features]

    def next(self):
        """
        This method is called for each bar of data (each time step).
        """
        # Get the features for the current time step
        current_features = self.features.iloc[len(self.data)-1:len(self.data)]

        # Make a prediction: 0=Sell, 1=Hold, 2=Buy
        prediction = self.model.predict(current_features)[0]

        if prediction == 2 and not self.position.is_long:
            # If the model predicts "Buy" and we don't have an open long position,
            # close any short position and buy.
            self.position.close()
            self.buy()
        elif prediction == 0 and not self.position.is_short:
            # If the model predicts "Sell" and we don't have an open short position,
            # close any long position and sell.
            self.position.close()
            self.sell()

def run_backtest(data_path, model_path):
    """
    Loads data and runs the backtest for the trained XGBoost model.
    """
    print("Loading data for backtest...")
    df = pd.read_parquet(data_path)

    # Backtesting.py requires a specific column naming convention:
    # Open, High, Low, Close, Volume. We rename our BTC columns to match.
    # We also pass the rest of the dataframe columns, which contain our features.
    backtest_df = df.rename(columns={
        'btc_open': 'Open',
        'btc_high': 'High',
        'btc_low': 'Low',
        'btc_close': 'Close',
        'btc_volume': 'Volume'
    })

    # We must use the test set for a fair evaluation.
    # The model was trained on the first 80%, so we test on the last 20%.
    split_index = int(len(backtest_df) * 0.8)
    test_data = backtest_df.iloc[split_index:]

    print("Starting backtest...")
    bt = Backtest(test_data, MLStrategy, cash=100_000, commission=.002)

    stats = bt.run()
    print("\n--- Backtest Results ---")
    print(stats)

    # The plot generation is currently disabled due to a version incompatibility
    # between backtesting.py and bokeh. The core statistical output is unaffected.
    # plot_filename = 'backtest_results.html'
    # bt.plot(filename=plot_filename, open_browser=False)
    # print(f"\nBacktest plot saved to {plot_filename}")

if __name__ == '__main__':
    DATA_PATH = 'data/processed/features_03_final.parquet'
    MODEL_PATH = 'src/tier1/modeling/xgboost_baseline_v1.json'
    run_backtest(DATA_PATH, MODEL_PATH)
