import os
from pathlib import Path

import click
import requests


def _rec_github_objects(language, branch=None):
    oid = 'null' if branch is None else f'"{branch}"'
    query = '''
    query {{
      search(first: 5, type: REPOSITORY, query: "stars:>1000 language:{language}") {{
        edges {{
          node {{
            ... on Repository {{
              name
              owner {{
                login
              }}
              object(expression: "master:", oid:{oid}) {{
                ... on Tree {{
                  entries {{
                    oid
                    name
                    type
                    object {{
                      ... on Blob {{
                        text
                        isTruncated
                        isBinary
                      }}
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    '''
    return query.format(language=language, oid=oid)


def _invoke_graphql_query(user, token, query):
    base_url = 'https://api.github.com/graphql'
    auth = requests.auth.HTTPBasicAuth(user, token)
    response = requests.post(base_url, auth=auth, json={'query': query})
    return response.json()['data']['search']['edges']


def _fetch_blobs(repositories, user, token, language, depth=1):
    trees = []
    blobs = []

    for l in range(depth):
        none_found = 0
        print(f'Depth {l} with {len(repositories)} repos...')
        for r in repositories:
            r = r['node']
            if r['object'] is not None:
                blobs += [(r['owner']['login'], r['name'], o) 
                          for o in r['object']['entries'] 
                          if o['type'] == 'blob' and o['name'].endswith('.py')]
                trees += [o for o in r['object']['entries'] 
                      if o['type'] == 'tree']
            else:
                none_found += 1

        print(f'Found {none_found} None objects out of {len(repositories)}')
        print(f'Obtaining new trees based on {len(trees)} parents...')
        new_repositories = [_invoke_graphql_query(
                                user=user, 
                                token=token, 
                                query=_rec_github_objects(language=language, 
                                                          branch=o['oid']))
                            for o in trees]

        repositories = [r for rl in new_repositories for r in rl]
        trees.clear()

    blobs = list({o[-1]['oid']: o for o in blobs}.values())
    return blobs


def _env(key):
    def f():
        return os.environ.get(key, '')
    return f


@click.command()
@click.option('--user', default=_env('GITHUB_USER'),
              help='GitHub user')
@click.option('--token', default=_env('GITHUB_TOKEN'), 
              help='Token credential associated with user')
@click.option('-o', '--output', required=True)
@click.option('--language', default='python')
def download(user, token, output, language):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)

    query = _rec_github_objects(language)
    repositories = _invoke_graphql_query(user, token, query)
    blobs = _fetch_blobs(repositories, user, token, language)

    for b in blobs:
        fname = f'{b[0]}_{b[1]}_{b[2]["name"]}]'
        with (output / fname).open('w') as f:
            f.write(b[2]["object"]["text"])


if __name__ == '__main__':
    download()
