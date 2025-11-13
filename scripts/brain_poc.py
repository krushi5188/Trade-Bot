import zmq
import time
import json
import uuid

def run_brain_poc():
    """
    This script simulates the AI Brain, acting as a ZeroMQ publisher.
    It sends a dummy trade signal every 5 seconds.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # The port must match the one the MT5 EA is listening on.
    # We use a high, non-standard port to avoid conflicts.
    port = 5555
    socket.bind(f"tcp://*:{port}")
    print(f"AI Brain (PoC) is publishing signals on port {port}...")

    while True:
        try:
            # Create a unique trade ID for each signal
            trade_id = str(uuid.uuid4())

            # Construct the trade signal as a Python dictionary
            signal = {
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "action": "BUY",
              "symbol": "EURUSD",
              "entry_price": 1.0550,
              "stop_loss": 1.0540,
              "take_profit": 1.0570,
              "trade_id": trade_id
            }

            # Convert the dictionary to a JSON string for transmission
            signal_json = json.dumps(signal)

            # Publish the signal
            print(f"Sending signal: {signal_json}")
            socket.send_string(signal_json)

            # Wait for 5 seconds before sending the next signal
            time.sleep(5)

        except KeyboardInterrupt:
            print("Shutting down AI Brain...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5) # Wait before retrying

    socket.close()
    context.term()

if __name__ == "__main__":
    run_brain_poc()
