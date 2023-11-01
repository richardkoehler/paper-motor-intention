"""Plot decoding times on brain surface."""
from __future__ import annotations

import os
import pathlib
import re
from typing import Literal

import matplotlib as mpl
import mne
import numpy as np
import pandas as pd
import trimesh
import trimesh.exchange.gltf
from matplotlib import axes, cm, colormaps, colors, figure
from matplotlib import pyplot as plt

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

DECODE = "decode"

CHANNEL = "ecog"
TIMES = (
    constants.RESULTS
    / DECODE
    / "stim_off_single_chs"
    / CHANNEL
    / "decodingtimes.csv"
)
SUBJECT_PICKS = ("paired", "all")
MEDICATION = ("OFF", "ON")

BASENAME = "decodingtimes_3Dplot"

ORIGIN = "auto"
_lh_views_dict = {
    "lateral": {"azimuth": 180.0, "elevation": 90.0, "focalpoint": ORIGIN},
    "medial": {"azimuth": 0.0, "elevation": 90.0, "focalpoint": ORIGIN},
    "rostral": {"azimuth": 90.0, "elevation": 90.0, "focalpoint": ORIGIN},
    "caudal": {"azimuth": 270.0, "elevation": 90.0, "focalpoint": ORIGIN},
    "ventral": {"azimuth": 180.0, "elevation": 180.0, "focalpoint": ORIGIN},
    "frontal": {"azimuth": 120.0, "elevation": 80.0, "focalpoint": ORIGIN},
    "parietal": {"azimuth": -120.0, "elevation": 60.0, "focalpoint": ORIGIN},
    "sagittal": {"azimuth": 180.0, "elevation": -90.0, "focalpoint": ORIGIN},
}
_rh_views_dict = {
    "lateral": {"azimuth": 180.0, "elevation": -90.0, "focalpoint": ORIGIN},
    "medial": {"azimuth": 0.0, "elevation": -90.0, "focalpoint": ORIGIN},
    "rostral": {"azimuth": -90.0, "elevation": -90.0, "focalpoint": ORIGIN},
    "caudal": {"azimuth": 90.0, "elevation": -90.0, "focalpoint": ORIGIN},
    "ventral": {"azimuth": 180.0, "elevation": 180.0, "focalpoint": ORIGIN},
    "frontal": {"azimuth": 60.0, "elevation": 80.0, "focalpoint": ORIGIN},
    "parietal": {"azimuth": -60.0, "elevation": 60.0, "focalpoint": ORIGIN},
    "sagittal": {"azimuth": 180.0, "elevation": -90.0, "focalpoint": ORIGIN},
}
_views = (
    {f"{view}_right": params for view, params in _rh_views_dict.items()}
    | {f"{view}_left": params for view, params in _lh_views_dict.items()}
    | {
        "dorsal": {"azimuth": 180.0, "elevation": 0.0, "focalpoint": ORIGIN},
        "axial": {
            "azimuth": 180.0,
            "elevation": 0.0,
            "focalpoint": ORIGIN,
            "roll": 0,
        },
        "coronal": {"azimuth": 90.0, "elevation": -90.0, "focalpoint": ORIGIN},
    }
)


def load_ch_pos_times(fpath: os.PathLike) -> tuple[np.ndarray, pd.Series]:
    coords = (
        pd.read_csv(fpath)
        .dropna(axis="columns", how="all")
        .drop(columns=["region", "used"])
        .rename(columns={"name": "Channel"})
        .set_index(["Subject", "Channel"])
    )
    times = (
        pd.read_csv(TIMES, index_col=["Subject", "Channel"])
        .query("Medication == 'OFF'")
        .loc[:, ["Earliest Timepoint"]]
        .rename(columns={"Earliest Timepoint": "Time [s]"})
    )
    coords = pd.concat([coords, times], axis="columns", join="inner")
    coords = coords.reset_index()
    coords["ch_id"] = coords["Subject"] + "-" + coords["Channel"]
    coords = coords.drop(columns=["Subject", "Channel"]).set_index("ch_id")
    coords["x"] = np.abs(coords["x"])
    ch_coords = coords[["x", "y", "z"]].to_numpy() * 1e-3
    return ch_coords, coords["Time [s]"]


def plot_ecog_3D_withcmap(
    ch_pos: np.ndarray | dict[str, np.ndarray] | None = None,
    values: np.ndarray | None = None,
    label: str = "Values",
    colormap: str = "viridis",
    vmin: float | Literal["auto"] = "auto",
    vmax: float | Literal["auto"] = "auto",
    project_to_surface: bool = False,
    template: Literal[
        "mni_icbm152_nlin_asym_09b"
    ] = "mni_icbm152_nlin_asym_09b",
    views: str | dict | list[str | dict] = "auto",
    figsize: Literal["auto"] | tuple[float, float] = "auto",
    outpath: pathlib.Path | None = None,
    show: bool = True,
    brain_kwargs: dict | None = None,
) -> None:
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    hemi = "both"

    if ch_pos is not None:
        if isinstance(ch_pos, dict):
            xyz = np.array(ch_pos.items())
            keys = ch_pos.keys()
        else:
            xyz = np.array(ch_pos)
            keys = (str(i) for i in range(xyz.shape[0]))
        if xyz.shape[1] != 3:
            msg = (
                "ch_pos must be an array with shape (n_channels, 3) or a dict "
                "with channel names as keys and items with the shape (3,). "
                f"Got: shape: {xyz.shape}"
            )
            raise ValueError(msg)
        mri_mni_trans = mne.read_talxfm(template, subjects_dir)
        mri_mni_inv = np.linalg.inv(mri_mni_trans["trans"])
        xyz_mri = mne.transforms.apply_trans(mri_mni_inv, xyz)
        if project_to_surface:
            path_mesh = subjects_dir / template / "surf" / f"{template}.glb"
            with open(path_mesh, "rb") as f:
                scene = trimesh.exchange.gltf.load_glb(f)
            mesh: trimesh.Trimesh = trimesh.Trimesh(
                **scene["geometry"]["geometry_0"]
            )
            xyz_mri = mesh.nearest.on_surface(xyz_mri)[0] * 1.03
        montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(keys, xyz_mri, strict=True)), coord_frame="mri"
        )
        info = mne.create_info(
            ch_names=montage.ch_names,
            sfreq=1000,
            ch_types="ecog",
            verbose=None,
        )
        info.set_montage(montage, verbose=False)
        identity = mne.transforms.Transform(
            fro="head", to="mri", trans=np.eye(4)
        )
        sensor_colors = None
        mapper = None
        if values is not None:
            cmap = colormaps[colormap]
            if vmin == "auto":
                vmin = values.min()
            if vmax == "auto":
                vmax = values.max()
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            sensor_colors = mapper.to_rgba(values)
        if brain_kwargs is None:
            brain_kwargs = {
                "surf": "pial",
                "cortex": "low_contrast",
                "alpha": 1.0,
                "background": "white",
            }
        brain = mne.viz.Brain(
            subjects_dir=subjects_dir,
            subject=template,
            hemi=hemi,
            show=False,
            block=False,
            **brain_kwargs,
        )
        brain.add_sensors(
            info,
            trans=identity,
            ecog=True,
            sensor_colors=sensor_colors,
        )
        if isinstance(views, str):
            if views == "auto":
                view_picks = ["dorsal", "lateral_right", "lateral_left"]
            else:
                view_picks = [views]
        elif isinstance(views, dict):
            view_picks = [views]
        else:
            if not isinstance(views, list):
                msg = (
                    "views must be either a string, a dictionary or a list"
                    f" of strings or dictionaries. Got {views}, {type(views)=}."
                )
                raise ValueError(msg)
            view_picks = views
        view_params = []
        for view in view_picks:
            if isinstance(view, str):
                try:
                    view_params.append(_views[view])
                except KeyError as err:
                    msg = f"View {view} not in {list(_views.keys())}"
                    raise ValueError(msg) from err
            elif isinstance(view, dict):
                view_params.append(view)
            else:
                msg = (
                    "views must be either a string, a dictionary or a list"
                    f" of strings or dictionaries. Got {view}, {type(view)=}."
                )
                raise ValueError(msg)
        if views == "auto":
            if figsize == "auto":
                figsize = (6.4, 4.8)
            fig = plt.figure(layout="constrained", figsize=figsize)
            left, right = fig.subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
            ax_left = left.add_subplot(111)
            axs_right = right.subplot_mosaic(
                """
                BD
                CD
                """,
                width_ratios=[5, 1],
            )
            axs = [ax_left] + [axs_right[item] for item in ("B", "C")]
            cax = axs_right["D"]
            cbar_kwargs = {
                "ax": cax,
                "fraction": 0.5,
                "shrink": 0.8,
            }
        else:
            if figsize == "auto":
                figsize = (6.2 * len(view_params) + 0.2, 4.8)
            width_ratios = [1] * len(view_params) + [0.03]
            fig, axs = plt.subplots(
                1,
                len(view_params) + 1,
                width_ratios=width_ratios,
                squeeze=True,
                figsize=figsize,
            )
            axs = [axs] if isinstance(axs, axes.Axes) else axs.tolist()
            cax = axs.pop(-1)
            cbar_kwargs = {"cax": cax}
        for ax, params in zip(axs, view_params, strict=True):
            brain.show_view(**params)
            brain.show()
            im = brain.screenshot(mode="rgb")
            nonwhite_pix = (im != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            im_cropped = im[nonwhite_row][:, nonwhite_col]
            ax.imshow(im_cropped)
            ax.set_axis_off()
        if values is not None:
            cbar = fig.colorbar(
                mapper,
                location="right",
                **cbar_kwargs,
            )
            cbar.ax.set_ylabel(label)
        if outpath is not None:
            print(outpath)
            save_fig(fig, outpath=outpath)
        if show:
            plt.show(block=True)
        else:
            plt.close(fig)


def save_fig(fig: figure.Figure, outpath: pathlib.Path) -> None:
    with mpl.rc_context({"pdf.fonttype": "TrueType", "svg.fonttype": "none"}):
        fig.savefig(str(outpath))
    if outpath.name.endswith(".svg"):
        with outpath.open(mode="r", encoding="utf-8") as f:
            svg_text = f.read()
        patched_svg = patch_affinity_svg(svg_text)
        with outpath.open(mode="w", encoding="utf-8") as f:
            f.write(patched_svg)


def patch_affinity_svg(svg_text: str) -> str:
    """Patch Matplotlib SVG so that it can be read by Affinity Designer."""
    matches = list(
        re.finditer(
            'font:( [0-9][0-9]?[0-9]?[0-9]?)? ([0-9.]+)px ([^;"]+)[";]',
            svg_text,
        )
    )
    if not matches:
        return svg_text
    svg_pieces = [svg_text[: matches[0].start()]]
    for i, match in enumerate(matches):
        # Change "font" style property to separate "font-size" and
        # "font-family" properties because Affinity ignores "font".
        font_weight, font_size_px, font_family = match.groups()
        new_font_style = (
            f"font-size: {float(font_size_px):.1f}px; "
            f"font-family: {font_family}"
        )
        if font_weight is not None:
            new_font_style = f"font-weight: {font_weight}; " + new_font_style
        svg_pieces.append(new_font_style)
        if i < len(matches) - 1:
            svg_pieces.append(
                svg_text[match.end() - 1 : matches[i + 1].start()]
            )
        else:
            svg_pieces.append(svg_text[match.end() - 1 :])
    return "".join(svg_pieces)


if __name__ == "__main__":
    motor_intention.plotting_settings.activate()
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["figure.dpi"] = 600
    height = 4.5 / 2.5  # cm to inch
    ch_pos, times = load_ch_pos_times(
        fpath=constants.DATA / "elec_ecog_bip.csv"
    )
    for descr, view in (
        ("lateral", {"azimuth": 185.0, "elevation": -50.0}),
        ("dorsal", "dorsal"),
    ):
        plot_ecog_3D_withcmap(
            ch_pos=ch_pos,
            values=times.to_numpy(),
            label=times.name,
            colormap="viridis_r",
            vmin=-2,
            vmax=2,
            project_to_surface=True,
            views=view,  # "auto",
            figsize=(height * 4 / 3, height),
            outpath=constants.PLOTS / f"{BASENAME}_{descr}.svg",
            show=True,
            brain_kwargs={
                "surf": "pial",
                "cortex": "low_contrast",
                "alpha": 1.0,
                "background": "white",
            },
        )
