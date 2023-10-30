"""Initialize settings for plotting."""
from __future__ import annotations

import os
import re
from enum import Enum

import cycler
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import figure
from matplotlib import pyplot as plt

# palette_colorblind_custom = (
#     (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
#     (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
#     (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
#     (0.8, 0.47058823529411764, 0.7372549019607844),
#     (0.8352941176470589, 0.3686274509803922, 0.0),
#     (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
#     (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
#     (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
#     (0.9254901960784314, 0.8823529411764706, 0.2),
#     (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
# )

PALETTE_WJN_2022 = np.array(
    (
        (55 / 255, 110 / 255, 180 / 255),  # blue - Med. OFF Stim. OFF
        (113 / 255, 188 / 255, 173 / 255),  # green - Med. ON Stim. OFF
        (223 / 255, 74 / 255, 74 / 255),  # red - Med. OFF Stim. ON
        (125 / 255, 125 / 255, 125 / 255),  # medium grey - STN
        (150 / 255, 150 / 255, 150 / 255),  # medium grey - Parietal Cortex
        (215 / 255, 178 / 255, 66 / 255),  # orange - Sensory Cortex
        (100 / 255, 100 / 255, 100 / 255),  # dark grey - Motor Cortex
        (215 / 255, 213 / 255, 203 / 255),  # medium/light grey
        (234 / 255, 234 / 255, 234 / 255),  # light grey
        # (64 / 255, 64 / 255, 64 / 255),
    )
)


class Color(Enum):
    ECOG_MEDOFF = (55 / 255, 110 / 255, 180 / 255)  # blue - Med. OFF Stim. OFF
    ECOG_MEDON = (113 / 255, 188 / 255, 173 / 255)  # green - Med. ON Stim. OFF
    ECOG_STIMON = (223 / 255, 74 / 255, 74 / 255)  # red - Med. OFF Stim. ON
    STN = (125 / 255, 125 / 255, 125 / 255)  # medium grey - STN
    PARIETAL = (
        150 / 255,
        150 / 255,
        150 / 255,
    )  # medium grey - Parietal Cortex
    SENSORY = (215 / 255, 178 / 255, 66 / 255)  # orange - Sensory Cortex
    MOTOR = (
        215 / 255,
        213 / 255,
        203 / 255,
    )  # medium/light grey - Motor Cortex


def activate() -> None:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 7

    mpl.rcParams["pdf.fonttype"] = "TrueType"
    mpl.rcParams["svg.fonttype"] = "none"

    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0
    mpl.rcParams["savefig.transparent"] = True

    mpl.rcParams["figure.figsize"] = [6.4, 4.8]

    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=PALETTE_WJN_2022)
    mpl.rcParams["axes.xmargin"] = 0

    mpl.rcParams["legend.title_fontsize"] = mpl.rcParams["legend.fontsize"]

    mpl.rcParams["lines.linewidth"] = 1


def ecogvsstn_medoff(show: bool = False) -> None:
    colors = [Color.ECOG_MEDOFF.value, Color.STN.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def ecogvsstn_medon(show: bool = False) -> None:
    colors = [Color.ECOG_MEDON.value, Color.STN.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def ecogvsstn_stimon(show: bool = False) -> None:
    colors = [Color.ECOG_STIMON.value, Color.STN.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def medoffvson(show: bool = False) -> None:
    colors = [Color.ECOG_MEDOFF.value, Color.ECOG_MEDON.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def stimoffvson(show: bool = False) -> None:
    colors = [Color.ECOG_MEDOFF.value, Color.ECOG_STIMON.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def medoff_medon_stimon(show: bool = False) -> None:
    colors = [
        Color.ECOG_MEDOFF.value,
        Color.ECOG_MEDON.value,
        Color.ECOG_STIMON.value,
    ]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def cortical_region(show: bool = False) -> None:
    colors = [Color.PARIETAL.value, Color.SENSORY.value, Color.MOTOR.value]
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)
    if show:
        sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
        plt.show(block=True)


def save_fig(fig: figure.Figure, outpath: str | os.PathLike) -> None:
    outpath = str(outpath)
    fig.savefig(outpath)  # , bbox_inches="tight")
    if outpath.endswith(".svg"):
        with open(outpath, encoding="utf-8") as f:
            svg_text = f.read()
        patched_svg = patch_affinity_svg(svg_text)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(patched_svg)


def patch_affinity_svg(svg_text: str) -> str:
    """Patch Matplotlib SVG so that it can be read by Affinity Designer."""
    matches = [
        x
        for x in re.finditer(
            'font:( [0-9][0-9]?[0-9]?[0-9]?)? ([0-9.]+)px ([^;"]+)[";]',
            svg_text,
        )
    ]
    if not matches:
        return svg_text
    svg_pieces = [svg_text[: matches[0].start()]]
    for i, match in enumerate(matches):
        # Change "font" style property to separate "font-size" and
        # "font-family" properties because Affinity ignores "font".
        group = match.groups()
        if len(group) == 2:
            font_weight, font_size_px, font_family = match.groups()
            new_font_style = (
                f"font-size: {float(font_size_px):.1f}px; "
                f"font-family: {font_family}"
            )
        else:
            font_weight, font_size_px, font_family = match.groups()
            if font_weight is not None:
                new_font_style = (
                    f"font-weight: {font_weight}; "
                    f"font-size: {float(font_size_px):.1f}px; "
                    f"font-family: {font_family}"
                )
            else:
                new_font_style = (
                    f"font-size: {float(font_size_px):.1f}px; "
                    f"font-family: {font_family}"
                )
        svg_pieces.append(new_font_style)
        if i < len(matches) - 1:
            svg_pieces.append(svg_text[match.end() - 1 : matches[i + 1].start()])
        else:
            svg_pieces.append(svg_text[match.end() - 1 :])
    return "".join(svg_pieces)


if __name__ == "__main__":
    activate()
    sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    plt.show(block=True)
