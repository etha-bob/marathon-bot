from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import discord
import matplotlib

matplotlib.use("Agg")  # headless backend for rendering before sending to Discord
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from discord import app_commands  # noqa: E402
from discord.ext import tasks  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN"]
APP_ID = 3065800  # Marathon
STEAM_URL = (
    f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={APP_ID}"
)
STEAMDB_URL = f"https://steamdb.info/app/{APP_ID}/charts/"
DISCORD_API = "https://discord.com/api/v10"
UPDATE_INTERVAL_MINUTES = 5
PEAK_FILE = Path(__file__).parent / "peak.json"
SAMPLES_FILE = Path(__file__).parent / "samples.json"
SAMPLE_RETENTION_HOURS = 24 * 7  # keep a week so we can extend windows later
DEFAULT_CHART_HOURS = 48
EMBED_COLOR = 0x1B2838
DESCRIPTION_MAX = 400  # Discord application description limit
TZ_EST = timezone(timedelta(hours=-5))
CHART_BG = "#1B2838"
CHART_LINE = "#A4D007"
CHART_AXIS = "#9BA3AF"
CHART_GRID = "#2A3F5A"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("marathon-bot")


class PeakStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.count: int = 0
        self.timestamp: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.count = int(data.get("count", 0))
            self.timestamp = data.get("timestamp")
        except (json.JSONDecodeError, ValueError, OSError) as e:
            log.warning("Could not parse peak file (%s); starting fresh", e)

    def update(self, players: int) -> bool:
        if players <= self.count:
            return False
        self.count = players
        self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.path.write_text(json.dumps({"count": self.count, "timestamp": self.timestamp}))
        return True


class SampleStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.samples: list[tuple[datetime, int]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.samples = [
                (datetime.fromisoformat(ts), int(count)) for ts, count in data
            ]
        except (json.JSONDecodeError, ValueError, OSError) as e:
            log.warning("Could not parse samples file (%s); starting fresh", e)
            self.samples = []

    def add(self, count: int) -> None:
        now = datetime.now(timezone.utc)
        self.samples.append((now, count))
        cutoff = now - timedelta(hours=SAMPLE_RETENTION_HOURS)
        self.samples = [(t, c) for t, c in self.samples if t >= cutoff]
        self._save()

    def _save(self) -> None:
        data = [[t.isoformat(timespec="seconds"), c] for t, c in self.samples]
        self.path.write_text(json.dumps(data))

    def window(self, hours: int) -> list[tuple[datetime, int]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [(t, c) for t, c in self.samples if t >= cutoff]


def render_chart(samples: list[tuple[datetime, int]], hours: int) -> bytes:
    times = [t for t, _ in samples]
    counts = [c for _, c in samples]
    peak = max(counts)
    minimum = min(counts)
    peak_idx = counts.index(peak)
    min_idx = counts.index(minimum)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    ax.plot(times, counts, color=CHART_LINE, linewidth=2.2)
    ax.fill_between(times, counts, color=CHART_LINE, alpha=0.12)

    ax.plot(times[peak_idx], peak, "o", color="white", markersize=6, zorder=5)
    ax.annotate(
        f"Peak {peak:,}",
        (times[peak_idx], peak),
        textcoords="offset points",
        xytext=(6, 8),
        color="white",
        fontsize=9,
    )
    ax.plot(times[min_idx], minimum, "o", color="white", markersize=6, zorder=5)
    ax.annotate(
        f"Min {minimum:,}",
        (times[min_idx], minimum),
        textcoords="offset points",
        xytext=(6, -14),
        color="white",
        fontsize=9,
    )

    ax.set_title(
        f"Marathon — players, last {hours}h  ·  times in EST",
        color="white",
        loc="left",
        fontsize=13,
        pad=12,
    )
    ax.tick_params(colors=CHART_AXIS, labelsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}")
    )
    for spine in ax.spines.values():
        spine.set_color(CHART_GRID)
    ax.grid(True, axis="y", alpha=0.35, color=CHART_GRID, linewidth=0.8)
    ax.grid(False, axis="x")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%b %d", tz=TZ_EST))
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


class MarathonBot(discord.Client):
    def __init__(self) -> None:
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)
        self.session: aiohttp.ClientSession | None = None
        self.peak = PeakStore(PEAK_FILE)
        self.samples = SampleStore(SAMPLES_FILE)
        self.last_count: int | None = None

    async def setup_hook(self) -> None:
        self.session = aiohttp.ClientSession()
        await self.tree.sync()
        self.update_presence.start()

    async def close(self) -> None:
        self.update_presence.cancel()
        if self.session is not None:
            await self.session.close()
        await super().close()

    async def fetch_player_count(self) -> int | None:
        assert self.session is not None
        try:
            async with self.session.get(
                STEAM_URL, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("Steam API request failed: %s", e)
            return None

        response = data.get("response") if isinstance(data, dict) else None
        if not response or response.get("result") != 1 or "player_count" not in response:
            log.warning("Unexpected Steam API payload: %s", data)
            return None
        return int(response["player_count"])

    async def update_application_description(self, count: int) -> None:
        assert self.session is not None
        now = datetime.now(TZ_EST)
        hour = now.hour % 12 or 12
        minute = now.strftime("%M")
        ampm = "AM" if now.hour < 12 else "PM"
        timestamp = now.strftime(f"%b %d %Y • {hour}:{minute} {ampm} EST")

        samples_24h = self.samples.window(24)
        peak_24h = max((c for _, c in samples_24h), default=count)

        lines = [
            f"🔥 Live count: {count:,}",
            f"📈 24 h peak: {peak_24h:,}",
            f"⏱  {timestamp}",
        ]
        description = "\n".join(lines)[:DESCRIPTION_MAX]

        try:
            async with self.session.patch(
                f"{DISCORD_API}/applications/@me",
                headers={
                    "Authorization": f"Bot {TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"description": description},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    log.warning(
                        "Failed to update bot description (%s): %s", resp.status, body
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("Could not update bot description: %s", e)

    @tasks.loop(minutes=UPDATE_INTERVAL_MINUTES)
    async def update_presence(self) -> None:
        count = await self.fetch_player_count()
        if count is None:
            return
        self.last_count = count
        self.samples.add(count)
        if self.peak.update(count):
            log.info("New peak recorded: %d", count)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"{count:,} Marathon players",
            )
        )
        await self.update_application_description(count)

    @update_presence.before_loop
    async def before_update_presence(self) -> None:
        await self.wait_until_ready()


bot = MarathonBot()


@bot.event
async def on_ready() -> None:
    user = bot.user
    log.info("Logged in as %s (id %s)", user, user.id if user else "?")


@bot.tree.command(
    name="players",
    description="Show the current Marathon player count on Steam.",
)
async def players_cmd(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True)
    count = await bot.fetch_player_count()
    if count is None:
        await interaction.followup.send(
            "Could not reach the Steam API right now. Try again in a moment."
        )
        return

    bot.last_count = count
    bot.peak.update(count)

    samples_24h = bot.samples.window(24)
    peak_24h = max((c for _, c in samples_24h), default=count)
    low_24h = min((c for _, c in samples_24h), default=count)

    embed = discord.Embed(
        title="Marathon — current players",
        url=STEAMDB_URL,
        color=EMBED_COLOR,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Current", value=f"{count:,}", inline=True)
    embed.add_field(name="24h Peak", value=f"{peak_24h:,}", inline=True)
    embed.add_field(name="24h Low", value=f"{low_24h:,}", inline=True)
    await interaction.followup.send(embed=embed)


@bot.tree.command(
    name="peak",
    description="Show the highest Marathon player count this bot has observed.",
)
async def peak_cmd(interaction: discord.Interaction) -> None:
    if not bot.peak.count:
        await interaction.response.send_message(
            "No peak recorded yet — give the bot a few minutes to poll Steam."
        )
        return
    if bot.peak.timestamp:
        peak_dt = datetime.fromisoformat(bot.peak.timestamp).astimezone(TZ_EST)
        hour = peak_dt.hour % 12 or 12
        ampm = "AM" if peak_dt.hour < 12 else "PM"
        when = peak_dt.strftime(f"%b %d %Y • {hour}:%M {ampm} EST")
    else:
        when = "an unknown time"
    embed = discord.Embed(
        title="Marathon — observed peak",
        url=STEAMDB_URL,
        description=f"**{bot.peak.count:,}** players, recorded at `{when}`.",
        color=EMBED_COLOR,
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(
    name="chart",
    description="Render a player count chart for a recent time window (default 48h).",
)
@app_commands.describe(hours="How many hours of history to include (1-168). Defaults to 48.")
async def chart_cmd(
    interaction: discord.Interaction,
    hours: app_commands.Range[int, 1, SAMPLE_RETENTION_HOURS] = DEFAULT_CHART_HOURS,
) -> None:
    await interaction.response.defer(thinking=True)
    samples = bot.samples.window(hours)
    if len(samples) < 2:
        await interaction.followup.send(
            f"Not enough data yet — only {len(samples)} sample(s) in the last {hours}h. "
            f"The bot polls every {UPDATE_INTERVAL_MINUTES} min, so the chart fills in over time."
        )
        return

    counts = [c for _, c in samples]
    peak = max(counts)
    minimum = min(counts)
    current = counts[-1]

    image_bytes = await asyncio.to_thread(render_chart, samples, hours)
    file = discord.File(io.BytesIO(image_bytes), filename=f"marathon-{hours}h.png")

    embed = discord.Embed(
        title=f"Marathon — last {hours}h",
        url=STEAMDB_URL,
        color=EMBED_COLOR,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Current", value=f"{current:,}")
    embed.add_field(name="Peak", value=f"{peak:,}")
    embed.add_field(name="Min", value=f"{minimum:,}")
    embed.set_image(url=f"attachment://marathon-{hours}h.png")
    embed.set_footer(text=f"{len(samples)} samples")

    await interaction.followup.send(embed=embed, file=file)


if __name__ == "__main__":
    bot.run(TOKEN)
