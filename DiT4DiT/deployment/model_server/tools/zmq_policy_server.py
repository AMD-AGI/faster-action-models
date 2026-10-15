# ZMQ version adapted for gr00t_wbc integration.

import logging
import traceback
from dataclasses import dataclass
from typing import Callable

import zmq

from . import msgpack_numpy


@dataclass
class EndpointHandler:
    """Handler for a server endpoint."""
    handler: Callable
    requires_input: bool = True


class ZmqPolicyServer:
    """Serves a policy using the ZMQ REQ/REP pattern.

    This is a ZMQ-based alternative to WebsocketPolicyServer, providing
    compatibility with gr00t_wbc's inference client pattern.

    Usage:
        policy = YourPolicyClass()
        server = ZmqPolicyServer(policy, host="0.0.0.0", port=5556)
        server.run()
    """

    def __init__(
        self,
        policy,
        host: str = "*",
        port: int = 5556,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self.running = True

        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

        # Register default endpoints
        self._endpoints: dict[str, EndpointHandler] = {}
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("get_action", self._handle_inference, requires_input=True)
        self.register_endpoint("predict_action", self._handle_inference, requires_input=True)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_metadata", self._get_metadata, requires_input=False)

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """Register a new endpoint handler."""
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _handle_ping(self) -> dict:
        """Handle ping request."""
        return {"status": "ok", "ok": True, "message": "Server is running"}

    def _kill_server(self) -> dict:
        """Handle kill request."""
        self.running = False
        return {"status": "ok", "ok": True, "message": "Server shutting down"}

    def _get_metadata(self) -> dict:
        """Return server metadata."""
        return {"status": "ok", "ok": True, "data": self._metadata}

    def _handle_inference(self, data: dict) -> dict:
        """Handle inference request by calling policy.predict_action()."""
        try:
            output = self._policy.predict_action(**data)
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "data": output,
            }
        except Exception as e:
            logging.exception("Policy inference error")
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "error": {
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            }

    def run(self):
        """Main server loop."""
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        logging.info(f"ZMQ Server is ready and listening on {addr}")
        print(f"ZMQ Server is ready and listening on {addr}")

        while self.running:
            try:
                # Receive request
                message = self.socket.recv()
                request = msgpack_numpy.unpackb(message)

                # Extract endpoint (default to "get_action")
                endpoint = request.get("endpoint", "get_action")

                # Route to handler
                if endpoint not in self._endpoints:
                    result = {
                        "status": "error",
                        "ok": False,
                        "error": {"message": f"Unknown endpoint: {endpoint}"},
                    }
                else:
                    handler = self._endpoints[endpoint]
                    if handler.requires_input:
                        data = request.get("data", {})
                        result = handler.handler(data)
                    else:
                        result = handler.handler()

                # Send response
                self.socket.send(msgpack_numpy.packb(result))

            except Exception as e:
                logging.exception(f"Error in ZMQ server: {e}")
                error_response = {
                    "status": "error",
                    "ok": False,
                    "error": {"message": str(e)},
                }
                try:
                    self.socket.send(msgpack_numpy.packb(error_response))
                except Exception:
                    pass

        # Cleanup
        self.socket.close()
        self.context.term()
        logging.info("ZMQ Server shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise NotImplementedError(
        "This module is not intended to be run directly. "
        "Use server_policy_zmq.py instead."
    )
