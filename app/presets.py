"""Fictional example games for the prediction page."""

PRESETS = {
    "Start from scratch": {},
    "🌾 Cozy farming sim with co-op": dict(
        name="Moonpetal Valley",
        short_description="Restore your grandmother's overgrown farm, befriend a village of spirits and grow magical crops alone or with up to four friends.",
        about=("Moonpetal Valley is a cozy farming and life sim set in a village where the seasons are ruled by "
               "friendly forest spirits. Clear the land, plant over 120 crops, raise animals and craft tools. "
               "Explore caves beneath the valley, fish in glowing rivers and decorate your farmhouse exactly the way you like. "
               "Get to know 30 villagers, each with their own stories, festivals and friendship events. "
               "Play solo or invite up to three friends in online co-op to build a farm together. "
               "Features: relaxing gameplay with no fail states, a hand-painted art style, full controller support, "
               "seasonal festivals, cooking and crafting, pet companions and a heart-warming story about family and home."),
        genres=["Indie", "Simulation", "RPG", "Casual"], price=19.99,
        categories=["Single-player", "Multi-player", "Co-op", "Online Co-op", "Steam Achievements",
                    "Steam Cloud", "Full controller support"],
        windows=1, mac=1, linux=1, achievements=45, n_languages=10, n_audio_languages=0, n_screenshots=14,
        has_website=1, month=9, dev_prior_games=1, dev_prior_best_reviews=400, self_published=1,
    ),
    "🕹️ First game from a solo dev (platformer)": dict(
        name="Pixel Jumper",
        short_description="A retro platformer about a small robot jumping through 30 levels.",
        about=("Pixel Jumper is a 2D pixel art platformer. Jump over spikes, avoid enemies and collect coins "
               "across 30 handmade levels in 3 worlds. Simple controls and a chiptune soundtrack. "
               "Can you reach the end?"),
        genres=["Indie", "Action", "Casual"], price=2.99,
        categories=["Single-player"], windows=1, mac=0, linux=0, achievements=0, n_languages=1,
        n_screenshots=5, has_website=0, month=6, dev_prior_games=0, dev_prior_best_reviews=0, self_published=1,
    ),
    "🔫 Free-to-play online shooter": dict(
        name="Overclock Arena",
        short_description="Fast team-based hero shooter. Pick one of 16 agents, master their abilities and fight in 5v5 online matches.",
        about=("Overclock Arena is a free-to-play team shooter where tactics meet speed. Choose from 16 agents, "
               "each with unique weapons and abilities, and battle across 10 maps in ranked and casual 5v5 modes. "
               "Climb the competitive ladder, join clans with friends, and unlock cosmetics through the battle pass. "
               "Dedicated servers, anti-cheat, cross-play with controller support and regular seasonal events keep every match fresh."),
        genres=["Action", "Free To Play", "Massively Multiplayer"], price=0.0,
        categories=["Multi-player", "PvP", "Online PvP", "Cross-Platform Multiplayer", "In-App Purchases",
                    "Steam Achievements", "Full controller support"],
        windows=1, mac=0, linux=0, required_age=17, achievements=60, n_languages=14, n_audio_languages=6,
        n_screenshots=12, has_website=1, month=3, dev_prior_games=2, dev_prior_best_reviews=3000,
        self_published=0, pub_prior_games=60, pub_prior_best_reviews=40000,
    ),
    "📖 Story-rich visual novel": dict(
        name="Letters to Autumn",
        short_description="A heartfelt visual novel about two pen pals whose letters start arriving from different years.",
        about=("Letters to Autumn is a narrative visual novel with multiple endings. Follow Mina and Theo as their "
               "letters cross time in mysterious ways. Every choice you make changes what they learn about each other. "
               "Fully illustrated scenes, an original piano soundtrack, 8 endings and around 6 hours of story."),
        genres=["Indie", "Adventure", "Casual"], price=9.99,
        categories=["Single-player", "Steam Achievements", "Steam Cloud"], windows=1, mac=1, linux=0,
        achievements=20, n_languages=3, n_screenshots=8, has_website=0, month=2,
        dev_prior_games=0, dev_prior_best_reviews=0, self_published=1,
    ),
}
