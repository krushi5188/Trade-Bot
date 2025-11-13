//+------------------------------------------------------------------+
//|                                              AI_Hands_EA.mq5 |
//|                      Copyright 2025, Your Company Name |
//|                                       https://www.example.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Company Name"
#property link      "https://www.example.com"
#property version   "1.00"

// --- Import the ZeroMQ library for MQL5
#include <Zmq/Zmq.mqh>

// --- Global variables for ZeroMQ
void* zmq_context;
void* zmq_socket;

// --- Input for the server address
input string ZmqServerHost = "127.0.0.1";
input int    ZmqServerPort = 5555; // MUST MATCH THE PYTHON SCRIPT

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // --- Initialize ZeroMQ context
    zmq_context = ZmqContext::New();
    if(zmq_context == NULL)
    {
        Print("Error creating ZeroMQ context.");
        return(INIT_FAILED);
    }

    // --- Create a SUBSCRIBER socket
    zmq_socket = ZmqSocket::New(zmq_context, ZMQ_SUB);
    if(zmq_socket == NULL)
    {
        Print("Error creating ZeroMQ socket.");
        ZmqContext::Term(zmq_context);
        return(INIT_FAILED);
    }

    // --- Connect to the Python "Brain" server
    string address = "tcp://" + ZmqServerHost + ":" + (string)ZmqServerPort;
    if(!ZmqSocket::Connect(zmq_socket, address))
    {
        Print("Error connecting to ZeroMQ server at ", address);
        ZmqSocket::Close(zmq_socket);
        ZmqContext::Term(zmq_context);
        return(INIT_FAILED);
    }

    // --- Subscribe to ALL messages. The filter is empty.
    ZmqSocket::SetSockOpt(zmq_socket, ZMQ_SUBSCRIBE, "");

    Print("AI Hands EA initialized. Listening for signals from ", address);

    // --- Use the OnTimer event to check for messages periodically
    EventSetTimer(1); // Check for a new message every 1 second (can be set to milliseconds)

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    if(zmq_socket != NULL)
        ZmqSocket::Close(zmq_socket);
    if(zmq_context != NULL)
        ZmqContext::Term(zmq_context);
    Print("AI Hands EA deinitialized.");
}

//+------------------------------------------------------------------+
//| Expert timer function                                            |
//+------------------------------------------------------------------+
void OnTimer()
{
    string message;
    // --- Check for a message WITHOUT WAITING (non-blocking)
    // --- This is the key to performance. The EA does not freeze.
    if(ZmqSocket::Recv(zmq_socket, message, ZMQ_DONTWAIT))
    {
        Print("Received signal from AI Brain: ", message);

        // --- TODO: Add JSON parsing and trade execution logic here ---
        // Example:
        // ParseJSONSignal(message);
        // ExecuteTrade(parsedSignal);
    }
}

// --- Example placeholder for trade execution logic
void ExecuteTrade(string signal)
{
    // Here you would parse the JSON and use the OrderSend() function
    // to place the trade with the correct SL/TP and price.
    // e.g., MqlTradeRequest request; MqlTradeResult result; ... OrderSend(request, result);
}
