"""Cache official card names/images for the selected deck files (no engine imports)."""
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote, urljoin, urlparse

ROOT = Path(__file__).resolve().parent
HOST = 'https://www.unionarena-tcg.com'


class CardPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.image = None
        self.name = ''
        self.number = ''
        self.capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if 'cardNameCol' in classes:
            self.capture = 'name'
        elif 'cardNumData' in classes:
            self.capture = 'number'
        if tag == 'img' and '/images/cardlist/card/' in attrs.get('src', ''):
            self.image = urljoin(HOST, attrs['src'])

    def handle_data(self, data):
        if self.capture:
            setattr(self, self.capture, getattr(self, self.capture) + data)

    def handle_endtag(self, tag):
        if tag in ('h2', 'span'):
            self.capture = None


def download(url, destination, delay):
    if urlparse(url).hostname != 'www.unionarena-tcg.com':
        raise ValueError('Only official UA URLs are permitted')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    time.sleep(delay)
    try:
        # Windows curl uses the system certificate store; TLS verification stays enabled.
        subprocess.run(['curl.exe' if __import__('os').name == 'nt' else 'curl',
                        '--fail', '--silent', '--show-error', '--retry', '2',
                        '--connect-timeout', '15', '--max-time', '60',
                        '--output', str(temporary), url], check=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_catalog(cards):
    catalog = {'schema_version': 1, 'preferred_language': 'tc',
               'updated_at': datetime.now(timezone.utc).isoformat(), 'cards': cards}
    payload = json.dumps(catalog, ensure_ascii=False, indent=2)
    for filename, content in [('card_catalog.json', payload + '\n'),
                              ('card_catalog.js', 'window.UA_CARD_CATALOG = ' +
                               payload.replace('<', '\\u003c') + ';\n')]:
        path = ROOT / 'data' / filename
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decks', nargs='+', default=['data/bleach_purple.json', 'data/ttgl_red.json'])
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--delay', type=float, default=0.5)
    args = parser.parse_args()
    if args.delay < 0.25:
        parser.error('--delay must be at least 0.25 seconds')
    cards = {}
    catalog_path = ROOT / 'data/card_catalog.json'
    if catalog_path.exists():
        cards = json.loads(catalog_path.read_text(encoding='utf-8'))['cards']
    requested = {}
    for filename in args.decks:
        deck = json.loads((ROOT / filename).read_text(encoding='utf-8'))
        for card in deck['cards']:
            requested[card['id']] = card['name']
    failures = []
    for index, (card_id, display_name) in enumerate(requested.items(), 1):
        try:
            if not re.fullmatch(r'[A-Za-z0-9-]+/[A-Za-z0-9-]+', card_id):
                raise ValueError(f'Invalid official card ID: {card_id}')
            slug = card_id.replace('/', '_')
            for language in ('tc', 'jp', 'en'):
                source = HOST + f'/{language}/cardlist/detail_iframe.php?card_no=' + quote(card_id, safe='')
                page = ROOT / 'data/card_pages' / (slug + '_' + language + '.html')
                if args.refresh or not page.exists():
                    download(source, page, args.delay)
                parsed = CardPage()
                parsed.feed(page.read_text(encoding='utf-8'))
                if parsed.number.strip() == card_id and parsed.name.strip() and parsed.image:
                    break
            else:
                raise ValueError(f'Official page did not match {card_id}')
            image_path = ROOT / 'assets/cards' / (slug + '.png')
            if args.refresh or not image_path.exists():
                download(parsed.image, image_path, args.delay)
            if not image_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
                image_path.unlink(missing_ok=True)
                raise ValueError('Response is not a PNG image')
            cards[card_id] = {'id': card_id, 'name': parsed.name.strip(),
                              'language': language,
                              'display_name': display_name,
                              'image': image_path.relative_to(ROOT).as_posix(),
                              'image_url': parsed.image, 'source': source}
            write_catalog(cards)
            print(f'[{index}/{len(requested)}] OK {card_id}', flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failures.append(card_id)
            print(f'FAILED {card_id}: {error}', flush=True)
    print(f'Completed: {len(requested) - len(failures)}/{len(requested)}; failures={failures}')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
