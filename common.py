import re


def feature_columns(columns, feature_select):
    try:
        feature, part = feature_select
    except ValueError:
        feature, part = (feature_select, None)

    if feature == 'hl':
        regex_filter = highlevel_regex()
    elif feature == 'genre':
        regex_filter = genre_regex()
    elif feature == 'mood':
        regex_filter = mood_regex()
    elif feature == 'voice':
        regex_filter = voice_regex()
    else:
        regex_filter = feature

    return filter_features(columns, regex_filter), part


def filter_features(columns, regex_filter):
    regex = re.compile(regex_filter)
    return list(filter(regex.match, columns))


def highlevel_regex():
    return r'highlevel\.\w+\.all\.\w+'


def genre_regex():
    return r'highlevel\.genre_tzanetakis\.all\.\w+'


def mood_regex():
    return r'highlevel\.mood_\w+\.all\.[a-z]+$'


def voice_regex():
    return r'(highlevel\.voice_instrumental\.all\.\w+|highlevel\.gender\.all\.\w+)'
