import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))

# Encoder Inputs
ax.add_patch(patches.Rectangle((1.0, 3.5), 1.5, 0.6, edgecolor='black', facecolor='#add8e6'))
ax.text(1.35, 3.75, "latent_input", fontsize=10)
ax.add_patch(patches.Rectangle((2.7, 3.5), 1.5, 0.6, edgecolor='black', facecolor='#add8e6'))
ax.text(3.0, 3.75, "proprio_input", fontsize=10)
ax.add_patch(patches.Rectangle((4.4, 3.5), 1.5, 0.6, edgecolor='black', facecolor='#90ee90'))
ax.text(4.65, 3.65, "image features \n       (src)", fontsize=10)

# Arrows from inputs to encoder
ax.annotate('', xy=(1.75, 3.1), xytext=(1.75, 3.5), arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(3.45, 3.1), xytext=(3.45, 3.5), arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(5.15, 3.1), xytext=(5.15, 3.5), arrowprops=dict(arrowstyle='->'))

# Encoder box
ax.add_patch(patches.Rectangle((1.45, 2.5), 4, 0.6, edgecolor='black', facecolor='#d3d3d3'))
ax.text(3.45, 2.7, "Transformer \nEncoder", fontsize=12, ha='center')

# memory output from encoder
ax.annotate('', xy=(3.45, 2.1), xytext=(3.45, 2.5), arrowprops=dict(arrowstyle='->'))

# Decoder box
ax.add_patch(patches.Rectangle((1.95, 1.5), 3.0, 0.6, edgecolor='black', facecolor='#ffe4b5'))
ax.text(3.5, 1.7, "Transformer \nDecoder", fontsize=12, ha='center')

# query_embed input (corrected direction)
ax.add_patch(patches.Rectangle((6.5, 3.5), 1.5, 0.6, edgecolor='black', facecolor='#f4cccc'))
ax.text(6.8, 3.75, "query_embed", fontsize=10)
ax.annotate('', xy=(7.25, 3.5), xytext=(4.93, 1.8), arrowprops=dict(arrowstyle='<-'))

# output from decoder
ax.annotate('', xy=(3.45, 1.5), xytext=(3.45, 1.1), arrowprops=dict(arrowstyle='<-'))
ax.add_patch(patches.Rectangle((2.55, 0.), 1.8, 0.6, edgecolor='black', facecolor='white'))
ax.text(3.2, 0.2, "Predicted object \ninformation", fontsize=12)

# Limits and formatting
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.axis('off')
plt.tight_layout()
plt.show()
