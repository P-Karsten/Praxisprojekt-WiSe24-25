import matplotlib.pyplot as plt
import io
import tensorflow as tf

def plot_reward_diagram(rewards, total_reward, step):
    labels = list(rewards.keys())
    values = list(rewards.values())
    percentages = [(v / total_reward * 100) if total_reward != 0 else 0 for v in values]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=['green' if v > 0 else 'red' for v in values])
    for bar, value, percent in zip(bars, values, percentages):
        height = bar.get_height()
        position = height + (0.05 * max(values)) if height >= 0 else height - (0.05 * abs(min(values)))
        ax.text(
            bar.get_x() + bar.get_width() / 2, position,
            f"{value:.2f}\n({percent:.1f}%)",
            ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=10, color='black'
        )
    ax.set_title(f"Rewards: (Step: {step}, Total: {total_reward:.2f})")
    ax.set_ylabel("Reward Value")
    ax.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    image = tf.image.decode_image(buf.getvalue(), channels=3)
    image = tf.expand_dims(image, axis=0)
    return image


def plot_action_chart(rewards, percentages, total_reward, step):
    labels = [f"{k} ({percent:.2f}%)" for k, percent in percentages.items()]
    values = [abs(v) for v in rewards.values()]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Executed actions: (Step: {step}, Total: {total_reward:.2f})")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    image = tf.image.decode_image(buf.getvalue(), channels=3)
    image = tf.squeeze(image, axis=0) if len(image.shape) == 4 else image
    image = tf.expand_dims(image, axis=0)
    return image