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


def check_valid_column(observation):
    """
        Validates that our observation has the minimum needed columns

    :param observation: json object
    :return:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """

    valid_columns = {
        "observation_id",
        "Type",
        "Date",
        "Part of a policing operation",
        "Latitude",
        "Longitude",
        "Gender",
        "Age range",
        "Officer-defined ethnicity",
        "Legislation",
        "Object of search",
        "station",
    }

    keys = set(observation.keys())

    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    return True, ""
