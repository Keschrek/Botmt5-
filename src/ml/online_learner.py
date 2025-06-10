import logging
from typing import Dict, Any, Optional, Tuple
from river import linear_model, preprocessing, optim, metrics, compose
# from sklearn.pipeline import Pipeline # Use river compose instead
import pandas as pd # Import pandas
import pickle # For saving/loading the model
from pathlib import Path # For file paths
import os # For creating directories

logger = logging.getLogger(__name__)

class OnlineLearner:
    def __init__(self, ml_cfg: Dict[str, Any]):
        self.ml_cfg = ml_cfg
        self.model = None # Placeholder for a river model, for example
        self.metric = None # Primary metric (e.g., Accuracy)
        self.metric_f1 = None # F1 Score
        self.metric_logloss = None # LogLoss
        self.model_path = ml_cfg.get('model_path', 'ml_model.pkl') # Path to save/load the model
        self.model_save_interval = ml_cfg.get('model_save_interval', 100) # Save model every N updates
        self._update_count = 0 # Counter for model updates
        # Store indicators config for feature extraction consistency
        self.indicators_cfg = ml_cfg.get('indicators', {})
        self.timezone = ml_cfg.get('timezone', 'UTC') # Get timezone from config for time features

        # Attempt to load the model if it exists
        if Path(self.model_path).exists():
             try:
                  self.load_model(self.model_path)
                  logger.info(f"Online learning model loaded from {self.model_path}.")
                  # Metrics are re-initialized after loading (see load_model)
             except Exception as e:
                  logger.error(f"Failed to load online learning model from {self.model_path}: {e}. Initializing a new model.")
                  self._initialize_model()
        else:
             logger.info("No existing online learning model found. Initializing a new model.")
             self._initialize_model()

    def _initialize_model(self):
        """Initializes a new online learning model based on ml_cfg."""
        try:
            # Define the preprocessing steps and model based on configuration
            pipeline_steps = []

            # Configure Scaler (example using StandardScaler)
            scaler_cfg = self.ml_cfg.get('scaler')
            if scaler_cfg and scaler_cfg.get('type') == 'StandardScaler':
                # StandardScaler works on numerical features. Need to define which features are numerical.
                # For now, let's assume all extracted features are numerical and scale them all.
                # In a real scenario, you might use a compose.ColumnTransformer.
                # For simplicity, applying scaler to all features in the pipeline.
                # If using a ColumnTransformer, you would define which columns to apply it to.
                pipeline_steps.append(('scaler', preprocessing.StandardScaler()))
                logger.info("Configured StandardScaler.")
            # Add more scaler types here if needed

            # Configure Model (example using LogisticRegression)
            model_cfg = self.ml_cfg.get('model')
            if model_cfg:
                model_type = model_cfg.get('type')
                if model_type == 'LogisticRegression':
                     # Extract learning_rate from model_cfg if available, otherwise from main ml_cfg
                    learning_rate = model_cfg.get('learning_rate', self.ml_cfg.get('learning_rate', 0.01))
                    pipeline_steps.append(('model', linear_model.LogisticRegression(optim.SGD(learning_rate))))
                    logger.info(f"Configured LogisticRegression with learning rate {learning_rate}.")
                # Add more model types here if needed (e.g., Perceptron, NaiveBayes, NN)
                # elif model_type == 'LinearRegression': ...
                # elif model_type == 'Perceptron': ...
                else:
                    logger.error(f"Unsupported model type specified in config: {model_type}")
                    raise ValueError("Unsupported model type")
            else:
                logger.error("No model configuration found in ml_cfg.")
                raise ValueError("No model configuration")

            if not pipeline_steps:
                 logger.error("No valid pipeline steps configured.")
                 raise ValueError("No valid pipeline steps")

            # Combine pipeline steps using compose.Pipeline
            self.model = compose.Pipeline(*pipeline_steps)
            
            # Initialize Metrics for classification
            self.metric = metrics.Accuracy() # Primary metric (Accuracy)
            self.metric_f1 = metrics.F1() # F1 Score
            self.metric_logloss = metrics.LogLoss() # LogLoss
            
            logger.info("New online learning model initialized.")

        except Exception as e:
            logger.error(f"Error initializing online learning model: {e}")
            self.model = None # Ensure model is None if initialization fails
            self.metric = None
            self.metric_f1 = None
            self.metric_logloss = None

    def process_trade_feedback(self, trade_data: Dict[str, Any]):
        """Processes feedback from a completed trade (e.g., P/L, outcome) to update the model."""
        logger.info(f"Processing trade feedback for trade: {trade_data.get('comment', 'N/A')}")

        if self.model is None:
             logger.warning("Cannot process trade feedback: Online learning model not initialized.")
             return

        try:
            # Extract features and target from trade_data
            # Assumes trade_data contains necessary information, including potentially the bar data at entry.
            features = self.extract_features(trade_data)
            target = self.extract_target(trade_data) # e.g., 1 for win, 0 for loss

            if features is None or target is None:
                 logger.warning("Skipping trade feedback processing: Could not extract features or target.")
                 return

            # Update the online learning model
            # river models learn one sample at a time
            xf, y = features, target # Use river's convention for features (x) and target (y)
            
            # Make a prediction BEFORE learning to evaluate the model's performance on this sample
            y_pred = None
            y_prob = None
            try:
                y_pred = self.model.predict_one(xf) # Predicted class (0 or 1)
                # Try to get probabilities if the model supports it (e.g., LogisticRegression)
                try:
                     y_prob = self.model.predict_proba_one(xf)
                except Exception as prob_e:
                     logger.debug(f"Model does not support predict_proba_one or failed: {prob_e}")
                     # y_prob remains None

                # Update metrics
                if self.metric:
                    self.metric.update(y, y_pred) # Update Accuracy
                    logger.debug(f"Accuracy updated: {self.metric.get()}")
                if self.metric_f1:
                     self.metric_f1.update(y, y_pred) # Update F1 Score
                     logger.debug(f"F1 Score updated: {self.metric_f1.get()}")
                if self.metric_logloss and y_prob is not None:
                     # LogLoss requires probabilities. Need probability of the true class y.
                     # The probabilities dict is like {class_value: probability}
                     true_class_prob = y_prob.get(y)
                     if true_class_prob is not None:
                          self.metric_logloss.update(y, y_prob) # Update LogLoss
                          logger.debug(f"LogLoss updated: {self.metric_logloss.get():.4f}")
                     else:
                          logger.warning(f"Could not get probability for true class {y} for LogLoss update.")

            except Exception as pred_e:
                 logger.warning(f"Prediction or metric update failed: {pred_e}. This might happen early in training.")
                 y_pred = None # Indicate prediction failed
                 y_prob = None # Indicate probabilities failed

            self.model.learn_one(xf, y) # Learn from the feedback
            logger.debug(f"Model learned one sample. Target: {y}, Predicted (before learning): {y_pred}")

            self._update_count += 1
            # Save model periodically
            if self._update_count % self.model_save_interval == 0:
                 self.save_model(self.model_path)
                 self._update_count = 0 # Reset counter after saving

        except Exception as e:
            logger.error(f"Error processing trade feedback: {e}")

    def extract_features(self, trade_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Helper method to extract features from trade data for the ML model training."""
        # Extract features from the entry bar data and trade metadata.

        entry_bar_data = trade_data.get('entry_bar_data')
        if not entry_bar_data:
             logger.warning("Cannot extract features from trade feedback: 'entry_bar_data' not found.")
             return None

        features = {}
        # Extract indicator values from entry_bar_data (assuming they are calculated and stored here)

        # EMA-Werte & -Differenz
        ema_fast_period = self.indicators_cfg.get('ema_fast_period', 8) # Default period
        ema_slow_period = self.indicators_cfg.get('ema_slow_period', 21) # Default period

        ema_fast_col = f'EMA_{ema_fast_period}'
        ema_slow_col = f'EMA_{ema_slow_period}'

        ema_fast = entry_bar_data.get(ema_fast_col)
        ema_slow = entry_bar_data.get(ema_slow_col)

        features[ema_fast_col] = ema_fast if ema_fast is not None else 0.0
        features[ema_slow_col] = ema_slow if ema_slow is not None else 0.0

        if ema_fast is not None and ema_slow is not None:
            features['ema_diff'] = ema_fast - ema_slow
        else:
            features['ema_diff'] = 0.0 # Impute difference if either EMA is missing

        # RSI
        rsi_period = self.indicators_cfg.get('rsi_period', 14)
        rsi_col = f'RSI_{rsi_period}'
        rsi_val = entry_bar_data.get(rsi_col)
        features[rsi_col] = rsi_val if rsi_val is not None else 50.0 # Impute RSI with neutral value

        # MACD-Histogramm
        macd_hist_col = 'MACD_hist' # Assuming column name
        macd_hist_val = entry_bar_data.get(macd_hist_col)
        features[macd_hist_col] = macd_hist_val if macd_hist_val is not None else 0.0 # Impute MACD Hist with 0

        # Bollinger-Abstand
        # Assumes entry_bar_data contains 'close', 'BB_upper', 'BB_lower'
        close_price = entry_bar_data.get('close')
        bb_upper = entry_bar_data.get('BB_upper')
        bb_lower = entry_bar_data.get('BB_lower')

        features['close'] = close_price if close_price is not None else entry_bar_data.get('open', 0.0) # Use open if close is missing
        features['BB_upper'] = bb_upper if bb_upper is not None else 0.0
        features['BB_lower'] = bb_lower if bb_lower is not None else 0.0

        if close_price is not None and bb_lower is not None:
            features['price_minus_lower'] = close_price - bb_lower
        else:
            features['price_minus_lower'] = 0.0

        if close_price is not None and bb_upper is not None:
            features['bb_upper_minus_price'] = bb_upper - close_price
        else:
            features['bb_upper_minus_price'] = 0.0

        # ATR
        atr_period = self.indicators_cfg.get('atr_period', 14)
        atr_col = f'ATR_{atr_period}'
        atr_val = entry_bar_data.get(atr_col)
        features[atr_col] = atr_val if atr_val is not None else 0.0 # Impute ATR with 0

        # Trade-Metadaten
        side = trade_data.get('side') # "Buy" or "Sell"
        features['side'] = 1 if side == "Buy" else (0 if side == "Sell" else 0.5) # Encode side: 1 for Buy, 0 for Sell, 0.5 for others/unknown

        features['risk_pct'] = trade_data.get('risk_pct', 0.0) # Risk per trade percentage
        features['entry_price'] = trade_data.get('entry_price', 0.0)

        # TP/SL distance (assuming absolute price difference)
        # TODO: Clarify if tp_distance/sl_distance are in Pips or absolute price difference and adjust accordingly.
        # Assuming absolute price difference for now.
        features['tp_distance'] = trade_data.get('tp_distance', 0.0)
        features['sl_distance'] = trade_data.get('sl_distance', 0.0)

        # Time features (optional)
        # entry_time = trade_data.get('entry_time') # Assuming entry_time is a datetime object or timestamp
        # if entry_time and pd.api.types.is_datetime64_any_dtype(entry_time) or isinstance(entry_time, (int, float)): # Check if it's a datetime or timestamp
        #      try:
        #           # Convert to datetime and localize if timezone is available
        #           if isinstance(entry_time, (int, float)): # If timestamp
        #                 dt_object = pd.to_datetime(entry_time, unit='s') # Assuming Unix timestamp
        #           else: # Assuming already a datetime object
        #                 dt_object = pd.to_datetime(entry_time)
        #
        #           if self.timezone and dt_object.tzinfo is None: # Localize if timezone is set and dt is naive
        #                 dt_object = dt_object.tz_localize('UTC').tz_convert(self.timezone) # Assume naive is UTC, then convert
        #           elif self.timezone and dt_object.tzinfo is not None: # Convert if already timezone-aware
        #                 dt_object = dt_object.tz_convert(self.timezone)
        #
        #           features['hour'] = dt_object.hour
        #           features['day_of_week'] = dt_object.dayofweek # Monday=0, Sunday=6
        #           # Add other time features like minute, quarter, day of month, month, year if needed
        #      except Exception as e:
        #           logger.warning(f"Could not extract time features from trade_data: {e}")
        # else:
        #      logger.debug("Entry time not available or not in expected format for time features.")

        # Ensure all feature values are numerical (float or int) for river models
        # Convert any non-numeric types if necessary. Imputation of None is done above.
        for key, value in list(features.items()): # Iterate over a copy if modifying dict during iteration
             if not isinstance(value, (int, float)):
                  try:
                       features[key] = float(value) # Attempt conversion
                  except (ValueError, TypeError):
                       logger.warning(f"Could not convert feature {key} value '{value}' to float. Imputing with 0.0.")
                       features[key] = 0.0
             if pd.isna(features[key]): # Check for pandas NaNs after potential conversion
                  features[key] = 0.0
                  logger.debug(f"Imputed NaN feature {key} with 0.0")

        logger.debug(f"Extracted features from trade data: {features}")
        return features if features else None # Return None if no features were extracted

    def extract_target(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """Helper method to extract the target variable (outcome) from trade data."""
        # Define and implement target extraction logic (e.g., binary classification: win/loss)
        # Target: 1 for profitable trade, 0 otherwise
        net_pl = trade_data.get('net_pl', None)
        if net_pl is None:
             logger.warning("Cannot extract target: 'net_pl' not found in trade feedback.")
             return None
             
        # For binary classification (win/loss)
        target = 1 if net_pl > 0 else 0
        logger.debug(f"Extracted target (win/loss): {target}")
        return target

    def extract_features_for_prediction(self, latest_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Helper method to extract features from latest market data for prediction."""
        if latest_data.empty:
             logger.warning("Cannot extract features for prediction: Latest data is empty.")
             return None

        # Extract features from the latest completed bar (last row)
        # Assumes latest_data is a pandas DataFrame with time index and indicator columns
        latest_bar = latest_data.iloc[-1].to_dict() # Convert the Series row to a dict

        features = {}
        # Extract the *same* set of features as used for training (extract_features method)
        # This is crucial: features for training and prediction must match.

        # EMA-Werte & -Differenz
        ema_fast_period = self.indicators_cfg.get('ema_fast_period', 8) # Default period
        ema_slow_period = self.indicators_cfg.get('ema_slow_period', 21) # Default period

        ema_fast_col = f'EMA_{ema_fast_period}'
        ema_slow_col = f'EMA_{ema_slow_period}'

        ema_fast = latest_bar.get(ema_fast_col)
        ema_slow = latest_bar.get(ema_slow_col)

        features[ema_fast_col] = ema_fast if ema_fast is not None else 0.0
        features[ema_slow_col] = ema_slow if ema_slow is not None else 0.0

        if ema_fast is not None and ema_slow is not None:
            features['ema_diff'] = ema_fast - ema_slow
        else:
            features['ema_diff'] = 0.0 # Impute difference if either EMA is missing

        # RSI
        rsi_period = self.indicators_cfg.get('rsi_period', 14)
        rsi_col = f'RSI_{rsi_period}'
        rsi_val = latest_bar.get(rsi_col)
        features[rsi_col] = rsi_val if rsi_val is not None else 50.0 # Impute RSI with neutral value

        # MACD-Histogramm
        macd_hist_col = 'MACD_hist' # Assuming column name
        macd_hist_val = latest_bar.get(macd_hist_col)
        features[macd_hist_col] = macd_hist_val if macd_hist_val is not None else 0.0 # Impute MACD Hist with 0

        # Bollinger-Abstand
        # Assumes latest_bar contains 'close', 'BB_upper', 'BB_lower'
        close_price = latest_bar.get('close')
        bb_upper = latest_bar.get('BB_upper')
        bb_lower = latest_bar.get('BB_lower')

        features['close'] = close_price if close_price is not None else latest_bar.get('open', 0.0) # Use open if close is missing
        features['BB_upper'] = bb_upper if bb_upper is not None else 0.0
        features['BB_lower'] = bb_lower if bb_lower is not None else 0.0

        if close_price is not None and bb_lower is not None:
            features['price_minus_lower'] = close_price - bb_lower
        else:
            features['price_minus_lower'] = 0.0

        if close_price is not None and bb_upper is not None:
            features['bb_upper_minus_price'] = bb_upper - close_price
        else:
            features['bb_upper_minus_price'] = 0.0

        # ATR
        atr_period = self.indicators_cfg.get('atr_period', 14)
        atr_col = f'ATR_{atr_period}'
        atr_val = latest_bar.get(atr_col)
        features[atr_col] = atr_val if atr_val is not None else 0.0 # Impute ATR with 0

        # Trade-Metadaten - Not applicable for prediction based on latest data, only for feedback.
        # Include placeholders with default values to maintain consistent feature set keys.
        features['side'] = 0.5 # Default/neutral value
        features['risk_pct'] = 0.0
        features['entry_price'] = 0.0
        features['tp_distance'] = 0.0
        features['sl_distance'] = 0.0

        # Time features (optional)
        # try:
        #      dt_object = latest_data.index[-1] # Get the timestamp from the index of the last row
        #      # Ensure it's a datetime object and localize/convert if timezone is set
        #      if isinstance(dt_object, (int, float)): # If timestamp in index
        #            dt_object = pd.to_datetime(dt_object, unit='s')
        #
        #      if self.timezone and dt_object.tzinfo is None: # Localize if timezone is set and dt is naive
        #            dt_object = dt_object.tz_localize('UTC').tz_convert(self.timezone) # Assume naive is UTC, then convert
        #      elif self.timezone and dt_object.tzinfo is not None: # Convert if already timezone-aware
        #            dt_object = dt_object.tz_convert(self.timezone)
        #
        #      features['hour'] = dt_object.hour
        #      features['day_of_week'] = dt_object.dayofweek # Monday=0, Sunday=6
        #      # Add other time features if needed
        # except Exception as e:
        #      logger.warning(f"Could not extract time features for prediction: {e}")

        # Ensure all feature values are numerical (float or int) and handle NaNs/None
        for key, value in list(features.items()): # Iterate over a copy if modifying dict during iteration
             if not isinstance(value, (int, float)):
                  try:
                       features[key] = float(value) # Attempt conversion
                  except (ValueError, TypeError):
                       logger.warning(f"Could not convert prediction feature {key} value '{value}' to float. Imputing with 0.0.")
                       features[key] = 0.0
             if pd.isna(features[key]): # Check for pandas NaNs after potential conversion
                  features[key] = 0.0
                  logger.debug(f"Imputed NaN prediction feature {key} with 0.0")

        logger.debug(f"Extracted prediction features: {features}")
        # Return None if no *indicator* features were extracted, even if metadata placeholders exist.
        # Adjust this logic if metadata features are considered critical for prediction.
        # For now, return features even if only placeholders exist, to maintain consistent keys.
        return features if features else None

    def predict(self, latest_data: pd.DataFrame) -> Optional[int]:
        """Uses the learned model to make a class prediction (e.g., 0 or 1) for the latest data."""
        if self.model is None:
             logger.warning("Cannot make prediction: Online learning model not initialized.")
             return None
             
        try:
            features = self.extract_features_for_prediction(latest_data)
            if features is None: # Check if features could be extracted
                 logger.warning("Skipping prediction: Could not extract features for prediction.")
                 return None

            # Use model.predict_one to get the class prediction
            # Ensure feature keys match the ones the model was trained on.
            # river models expect features as dictionaries.
            prediction = self.model.predict_one(features)
            logger.debug(f"Prediction: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None # Return None if prediction fails


    def get_confidence_score(self, latest_data: pd.DataFrame) -> float:
        """Uses the learned model to provide a confidence score (probability of the positive class, e.g., win)."""
        if self.model is None:
             logger.warning("Cannot get confidence score: Online learning model not initialized.")
             return 0.5 # Default to 0.5 if model is not available

        try:
            features = self.extract_features_for_prediction(latest_data)
            if features is None: # Check if features could be extracted
                 logger.warning("Skipping confidence score: Could not extract features for prediction.")
                 return 0.5 # Default confidence if features cannot be extracted

            # Use model.predict_proba_one to get class probabilities
            # This method returns a dictionary like {class_value: probability}
            probabilities = self.model.predict_proba_one(features)
            
            # For binary classification (target 0 or 1), get probability of the positive class (1)
            confidence = probabilities.get(1, 0.5) # Default to 0.5 if class 1 not in probabilities
            
            logger.debug(f"Confidence score (prob of 1): {confidence:.2f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting confidence score: {e}")
            return 0.5 # Default to 0.5 if calculation fails

    def save_model(self, path: str):
        """Saves the current state of the online learning model."""
        if self.model is None:
            logger.warning("Cannot save model: Model is not initialized.")
            return

        try:
            model_dir = Path(path).parent
            os.makedirs(model_dir, exist_ok=True) # Ensure directory exists

            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Online learning model saved to {path}. Diffusion-enabled.")
        except Exception as e:
            logger.error(f"Failed to save online learning model to {path}: {e}")

    def load_model(self, path: str):
        """Loads the state of the online learning model from a file."""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Online learning model loaded from {path}.")
            # Re-initialize metrics after loading the model
            # TODO: Make metrics configurable and load/save their state if necessary
            self.metric = metrics.Accuracy() # Assuming Accuracy metric is always used
            self.metric_f1 = metrics.F1()
            self.metric_logloss = metrics.LogLoss()
        except FileNotFoundError:
            logger.warning(f"Model file not found at {path}.")
            self.model = None
            self.metric = None
            self.metric_f1 = None
            self.metric_logloss = None
            raise # Re-raise to indicate loading failed and new model might be needed
        except Exception as e:
            logger.error(f"Failed to load online learning model from {path}: {e}")
            self.model = None
            self.metric = None
            self.metric_f1 = None
            self.metric_logloss = None
            raise # Re-raise to indicate loading failed

# Note: This module will be integrated into LiveTrading to receive trade feedback and potentially influence trade decisions. 