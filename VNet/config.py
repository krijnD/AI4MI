from typing import Any

class params:

	datasets_params: dict[str, dict[str, Any]] = {}
	datasets_params["TOY2"] = {'K': 2, 'net': None, 'B': 2}
	datasets_params["SEGTHOR"] = {'K': 5, 'net': None, 'B': 8}