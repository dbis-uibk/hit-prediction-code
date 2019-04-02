import re


def feature_columns(columns, feature):
    if feature == 'hl':
        regex_filter = highlevel_regex()
    else:
        regex_filter = feature

    return filter_features(columns, regex_filter)


def filter_features(columns, regex_filter):
    regex = re.compile(regex_filter)
    return list(filter(regex.search, columns))


def highlevel_regex():
    return r'highlevel\.\w+\.all\.\w+'
