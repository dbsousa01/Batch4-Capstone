"""
    Functions to validate the input json
"""


def check_request(request):
    """
    Validates that our request is well formatted

    Returns:
    - assertion value: True if request is ok, False otherwise
    - error message: empty if request is ok, False otherwise
    """

    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error

    return True, ""
