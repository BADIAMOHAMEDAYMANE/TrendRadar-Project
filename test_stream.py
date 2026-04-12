from src.collect import stream_posts

print("Démarrage du test streaming...")
count = 0
for post in stream_posts(
    subreddits="Football+Euro",
    keywords="Sport",       # pas de filtre
    max_posts=5,         # on veut juste 5 posts
    poll_interval=10,
):
    count += 1
    print(f"[{count}] {post['subreddit']} | {post['title'][:60]}")

print(f"\n✅ {count} posts reçus")