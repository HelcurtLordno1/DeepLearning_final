"""Draw the two final architectures without constructing or training a model."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BLUE='#add8f5'
TEAL='#b8dfd5'
WHITE='#f4f6f8'
GOLD='#ffe0a5'
INK='#243746'


def canvas(height):
    fig,ax=plt.subplots(figsize=(8.5,height))
    ax.set(xlim=(0,10),ylim=(0,height),aspect='auto')
    ax.axis('off')
    return fig,ax


def box(ax,x,y,w,h,text,color=WHITE,size=10.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.035',
                              linewidth=1.1,edgecolor=INK,facecolor=color,zorder=3))
    ax.text(x,y,text,ha='center',va='center',fontsize=size,color=INK,zorder=4)


def arrow(ax,points,label=None,dashed=False,color=INK):
    for i,(a,b) in enumerate(zip(points,points[1:])):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>' if i==len(points)-2 else '-',
                    mutation_scale=11,linewidth=1.1,color=color,
                    linestyle='--' if dashed else '-',zorder=2))
    if label:
        ax.text(*label[:2],label[2],fontsize=8.5,ha='center',va='center',color=color,
                bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=5)


def draw_task2_architecture():
    fig,ax=canvas(7.6)
    box(ax,5,7.13,4.2,.48,'48 hourly observations: X [B,48,7]',BLUE,11)
    box(ax,5,6.35,4.2,.68,'LSTM encoder (7 → 128)\nMemory [B,48,128]; final states h, c',TEAL,11)
    box(ax,5,5.44,4.2,.55,'Repeat final hidden state 5 times\nDecoder inputs [B,5,128]',WHITE,11)
    box(ax,5,4.56,4.2,.62,'LSTM decoder (128 → 128)\nDecoded states D [B,5,128]',TEAL,11)
    box(ax,5,3.62,4.2,.64,'Scaled dot-product attention\nQueries: D; keys/values: encoder memory\nAttended context C [B,5,128]',BLUE,10.5)
    box(ax,5,2.72,4.2,.51,'Concatenate [D, C] → [B,5,256]',WHITE,11)
    box(ax,5,1.9,4.2,.62,'Dropout (p=0) + shared Linear(256 → 1)\nSqueeze → corrections Δ [B,5]',WHITE,11)
    ax.add_patch(Circle((5,1.13),.17,edgecolor=INK,facecolor=GOLD,zorder=3));ax.text(5,1.13,'+',ha='center',va='center',fontsize=14,zorder=4)
    box(ax,5,.48,4.2,.45,'Five OT forecasts: [B,5]',GOLD,11)
    box(ax,1.17,1.88,1.95,.66,'Last observed OT\n[B], broadcast\nto all 5 horizons',GOLD,9.5)
    for a,b in [(6.89,6.69),(6.01,5.72),(5.165,4.87),(4.25,3.94),(3.30,2.975),(2.465,2.21),(1.59,1.30),(0.96,.705)]:arrow(ax,[(5,a),(5,b)])
    arrow(ax,[(2.9,6.35),(.25,6.35),(.25,3.62),(2.9,3.62)],(.9,4.15,'48 encoder\npositions'),color='#287aa4')
    arrow(ax,[(7.1,6.35),(9.55,6.35),(9.55,4.70),(7.1,4.70)],(8.45,5.7,'Initialize decoder\nwith h and c'),True,'#72578b')
    arrow(ax,[(7.1,4.40),(8.65,4.40),(8.65,2.72),(7.1,2.72)],(8.65,3.20,'D: direct path'))
    arrow(ax,[(1.17,1.55),(1.17,1.13),(4.83,1.13)])
    ax.text(5,-.06,'B = batch size. Inverse scaling restores the OT scale before scoring.\nSolid arrows: tensor flow. Dashed arrow: recurrent-state initialization.',ha='center',va='top',fontsize=8.5)
    fig.tight_layout(pad=.3)
    return fig


def draw_task3_architecture():
    fig,ax=canvas(10.8)
    fig.set_size_inches(8.5,9.1)
    box(ax,5,10.48,4.8,.43,'RGB crop [B,3,64,192]',BLUE,11)
    boxes=[(9.76,'Conv 3→32, 3×3, pad 1 → ReLU → MaxPool 2×2\n[B,32,32,96]'),
           (8.93,'Conv 32→64, 3×3, pad 1 → ReLU → MaxPool 2×2\n[B,64,16,48]'),
           (8.10,'Conv 64→128, 3×3, pad 1 → ReLU → MaxPool (2,1)\nF [B,128,8,48]')]
    for y,text in boxes:box(ax,5,y,6.35,.61,text,BLUE,10.5)
    arrow(ax,[(5,10.265),(5,10.065)])
    arrow(ax,[(5,9.455),(5,9.235)]);arrow(ax,[(5,8.625),(5,8.405)])
    box(ax,7.45,7.17,4.45,.63,'Refinement: Conv128→128 → ReLU\n→ Conv128→128 (both 3×3, pad 1)\nMultiply branch output by 0.1',BLUE,9.5)
    ax.add_patch(Circle((3.2,7.17),.17,edgecolor=INK,facecolor=GOLD,zorder=3));ax.text(3.2,7.17,'+',ha='center',va='center',fontsize=14,zorder=4)
    arrow(ax,[(3.2,7.795),(3.2,7.34)],(2.25,7.58,'Identity F'))
    arrow(ax,[(7.45,7.795),(7.45,7.485)]);arrow(ax,[(5.225,7.17),(3.37,7.17)])
    box(ax,5,6.30,6.1,.68,'Permute to columns + flatten height: [B,48,1024]\nLinear(1024 → 128) + learned column positions\nMemory M [B,48,128]',WHITE,10.5)
    arrow(ax,[(3.2,7),(3.2,6.82),(5,6.82),(5,6.64)])
    box(ax,3.25,5.30,3.0,.58,'Mean of 48 columns\nGlobal context g [B,128]',WHITE,10.5)
    box(ax,7.05,5.30,3.45,.58,'AdaptiveAvgPool1d(7)\nLocal regions R [B,7,128]',WHITE,10.5)
    arrow(ax,[(3.25,5.96),(3.25,5.59)]);arrow(ax,[(7.05,5.96),(7.05,5.59)])
    box(ax,5,4.36,4.85,.56,'0.5 × repeated g + 0.5 × R\nSeven decoder inputs [B,7,128]',GOLD,11)
    arrow(ax,[(3.25,5.01),(3.25,4.84),(4.1,4.84),(4.1,4.64)])
    arrow(ax,[(7.05,5.01),(7.05,4.84),(5.9,4.84),(5.9,4.64)])
    box(ax,5,3.47,4.85,.56,'GRU decoder (128 → 128)\nDecoded states D [B,7,128]',TEAL,11)
    arrow(ax,[(5,4.08),(5,3.75)])
    arrow(ax,[(1.75,5.30),(1.1,5.30),(1.1,3.47),(2.575,3.47)],(1.05,4.26,'g initializes\nGRU state'),True,'#72578b')
    box(ax,5,2.57,4.85,.62,'Scaled dot-product attention\nQueries: D; keys/values: M\nContext C [B,7,128]; weights [B,7,48]',BLUE,10.5)
    arrow(ax,[(5,3.19),(5,2.88)])
    arrow(ax,[(1.95,6.30),(.22,6.30),(.22,2.57),(2.575,2.57)],(.4,2.95,'M'),color='#287aa4')
    box(ax,5,1.69,4.85,.49,'Concatenate [D, C] → [B,7,256]',WHITE,11)
    arrow(ax,[(5,2.26),(5,1.935)])
    arrow(ax,[(7.425,3.47),(9.75,3.47),(9.75,1.69),(7.425,1.69)],(9.55,2.01,'D'))
    box(ax,5,.82,5.7,.62,'Shared Linear(256 → 67) → logits [B,7,67]\nArgmax + vocabulary → 7-character plate',GOLD,11)
    arrow(ax,[(5,1.445),(5,1.13)])
    ax.text(5,.10,'Blue: CNN / attention. Green: recurrent decoder. Orange: combination / output.\nAll 7 positions, including the province character, are retained.',ha='center',fontsize=8.5)
    fig.tight_layout(pad=.3)
    return fig
