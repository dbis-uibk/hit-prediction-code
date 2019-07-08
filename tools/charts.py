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
    charts = billboard.ChartData('hot-100')
    end_date = charts.date
    known_songs = set()

    count = 1
    entries = []
    while charts.previousDate:
        for hit in charts:
            if str(hit) not in known_songs:
                known_songs.add(str(hit))
                entries.append(hit.__dict__)

        start_date = charts.date
        print(start_date, 'week', count)
        charts = billboard.ChartData('hot-100', charts.previousDate)

        if (count % 52 == 1) and (start_date != end_date):
            bb_hits = pd.DataFrame(entries)
            bb_hits.to_csv(RESULT_PATH + '/billboard_100_' + str(start_date) +
                           '_' + str(end_date) + '.csv')

        count += 1

    bb_hits = pd.DataFrame(entries)
    bb_hits.to_csv(RESULT_PATH + '/billboard_100_' + str(start_date) + '_' +
                   str(end_date) + '.csv')


if __name__ == '__main__':
    cli()
