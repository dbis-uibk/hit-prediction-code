import billboard

import click

import pandas as pd

RESULT_PATH = '.'


@click.group()
@click.option(
    '--path', default='.', help='The path where the results are stored.')
def cli(path):
    global RESULT_PATH
    RESULT_PATH = path


@cli.command()
def billboard_100():
    load_billboard_charts('hot-100')


@cli.command()
def billboard_200():
    load_billboard_charts('billboard-200')


def load_billboard_charts(name):
    charts = billboard.ChartData(name)
    end_date = charts.date
    known_songs = set()

    count = 1
    entries = []
    while charts.previousDate:
        for hit in charts:
            if str(hit) not in known_songs:
                known_songs.add(str(hit))
                entry = {
                    **hit.__dict__,
                    'latest-date': charts.date,
                }
                entries.append(entry)

        start_date = charts.date
        print(start_date, 'week', count)
        charts = billboard.ChartData(name, charts.previousDate)

        if (count % 52 == 1) and (start_date != end_date):
            bb_hits = pd.DataFrame(entries)
            bb_hits.to_csv(RESULT_PATH + '/billboard_' + name + '_' +
                           str(start_date) + '_' + str(end_date) + '.csv')

        count += 1

    bb_hits = pd.DataFrame(entries)
    bb_hits.to_csv(RESULT_PATH + '/billboard_' + name + '_' + str(start_date) +
                   '_' + str(end_date) + '.csv')


if __name__ == '__main__':
    cli()
