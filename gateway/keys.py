"""
Create, list and revoke gateway API keys. Redis stores only a hash of each key.

Run:
    python -m gateway.keys create alice --rpm 60 --tpm 20000
    python -m gateway.keys list
    python -m gateway.keys revoke alice
"""

import argparse
import secrets

import redis

from gateway.app import REDIS_URL, key_hash


def main():
    p = argparse.ArgumentParser(description="TokenForge gateway API keys")
    sub = p.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create", help="make a key and print it once")
    create.add_argument("name")
    create.add_argument("--rpm", type=int, default=60, help="requests per minute")
    create.add_argument("--tpm", type=int, default=20000, help="output tokens per minute")
    sub.add_parser("list")
    revoke = sub.add_parser("revoke", help="delete every key with this name")
    revoke.add_argument("name")
    args = p.parse_args()

    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    if args.cmd == "create":
        key = "tf-" + secrets.token_urlsafe(32)
        r.hset(f"key:{key_hash(key)}", mapping={"name": args.name, "rpm": args.rpm, "tpm": args.tpm})
        print(key)
        return
    for k in r.scan_iter("key:*"):
        info = r.hgetall(k)
        if args.cmd == "list":
            print(f"{info['name']:<20} rpm={info['rpm']:<6} tpm={info['tpm']:<8} {k[4:16]}…")
        elif info["name"] == args.name:
            r.delete(k)
            print(f"revoked {k[4:16]}…")


if __name__ == "__main__":
    main()
