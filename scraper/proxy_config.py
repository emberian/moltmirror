#!/usr/bin/env python3
"""
Proxy Configuration for Moltbook Scraping
Routes all scraping through a configured proxy to avoid IP blocking
"""

import os
import aiohttp
import requests
from typing import Optional, Dict, Any

# Proxy configuration from environment variables
# Set these in your .env file or docker-compose.yml
_raw_proxy_url = os.getenv("SCRAPER_PROXY_URL", "")  # e.g., "http://user:pass@proxy:8080" or "socks5://proxy:1080"

# When running in Docker, localhost references need to be translated to host.docker.internal
def _translate_proxy_url(url: str) -> str:
    """Translate localhost references for Docker compatibility"""
    if not url:
        return url
    # Replace 127.0.0.1 or localhost with host.docker.internal when in Docker
    in_docker = os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER', '')
    if in_docker:
        url = url.replace('127.0.0.1', 'host.docker.internal')
        url = url.replace('localhost', 'host.docker.internal')
    return url

PROXY_URL = _translate_proxy_url(_raw_proxy_url)
PROXY_ENABLED = os.getenv("SCRAPER_PROXY_ENABLED", "true").lower() in ("true", "1", "yes")

# Alternative: per-protocol proxies
HTTP_PROXY = os.getenv("SCRAPER_HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("SCRAPER_HTTPS_PROXY", "")
SOCKS_PROXY = os.getenv("SCRAPER_SOCKS_PROXY", "")  # e.g., "socks5://user:pass@host:port"


def get_proxy_url() -> Optional[str]:
    """Get the configured proxy URL"""
    if not PROXY_ENABLED:
        return None

    if PROXY_URL:
        return PROXY_URL

    if HTTPS_PROXY:
        return HTTPS_PROXY

    if HTTP_PROXY:
        return HTTP_PROXY

    if SOCKS_PROXY:
        return SOCKS_PROXY

    return None


def get_requests_proxies() -> Optional[Dict[str, str]]:
    """Get proxy dict for the requests library"""
    proxy = get_proxy_url()
    if not proxy:
        return None

    # requests library format
    return {
        "http": proxy,
        "https": proxy
    }


def get_aiohttp_connector() -> aiohttp.TCPConnector:
    """Get aiohttp connector (for non-SOCKS proxies, use proxy param on request)"""
    return aiohttp.TCPConnector(
        limit=50,
        limit_per_host=10,
        force_close=True
    )


def get_aiohttp_proxy() -> Optional[str]:
    """Get proxy URL for aiohttp requests"""
    proxy = get_proxy_url()

    # aiohttp doesn't support SOCKS directly - need aiohttp-socks for that
    if proxy and proxy.startswith("socks"):
        print(f"Warning: aiohttp needs aiohttp-socks for SOCKS proxies. Install with: pip install aiohttp-socks")
        return None

    return proxy


def create_requests_session() -> requests.Session:
    """Create a requests session with proxy configured"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'MoltmirrorSync/1.0 (research archiver)',
        'Accept': 'application/json',
    })

    proxies = get_requests_proxies()
    if proxies:
        session.proxies.update(proxies)
        print(f"[Proxy] Using proxy: {mask_proxy_url(proxies.get('https', ''))}")

    return session


def mask_proxy_url(url: str) -> str:
    """Mask credentials in proxy URL for logging"""
    if not url:
        return "(none)"

    # Mask password if present
    if "@" in url and ":" in url.split("@")[0]:
        protocol_user = url.split("@")[0]
        rest = "@".join(url.split("@")[1:])
        if "://" in protocol_user:
            protocol, user_pass = protocol_user.split("://")
            if ":" in user_pass:
                user = user_pass.split(":")[0]
                return f"{protocol}://{user}:****@{rest}"
        return f"****@{rest}"

    return url


async def create_aiohttp_session() -> aiohttp.ClientSession:
    """Create an aiohttp session with proxy configured"""
    proxy = get_aiohttp_proxy()

    # Try to use aiohttp-socks if SOCKS proxy is configured
    socks = get_proxy_url()
    if socks and socks.startswith("socks"):
        try:
            from aiohttp_socks import ProxyConnector
            connector = ProxyConnector.from_url(socks)
            print(f"[Proxy] Using SOCKS proxy: {mask_proxy_url(socks)}")
            return aiohttp.ClientSession(
                connector=connector,
                headers={
                    'User-Agent': 'MoltmirrorSync/1.0 (research archiver)',
                    'Accept': 'application/json',
                }
            )
        except ImportError:
            print("Warning: aiohttp-socks not installed. Install with: pip install aiohttp-socks")

    # Regular HTTP/HTTPS proxy or no proxy
    connector = get_aiohttp_connector()
    session = aiohttp.ClientSession(
        connector=connector,
        headers={
            'User-Agent': 'MoltmirrorSync/1.0 (research archiver)',
            'Accept': 'application/json',
        }
    )

    if proxy:
        print(f"[Proxy] Using HTTP proxy: {mask_proxy_url(proxy)}")

    return session


def get_proxy_status() -> Dict[str, Any]:
    """Get current proxy configuration status"""
    return {
        'enabled': PROXY_ENABLED,
        'proxy_url': mask_proxy_url(get_proxy_url() or ""),
        'type': 'socks' if (get_proxy_url() or "").startswith("socks") else 'http',
        'configured': bool(get_proxy_url())
    }


if __name__ == "__main__":
    status = get_proxy_status()
    print("Proxy Configuration:")
    for k, v in status.items():
        print(f"  {k}: {v}")
