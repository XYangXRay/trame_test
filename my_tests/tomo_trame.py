from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from trame.app import get_server
from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vtk, vuetify, html

from trame_vtk.modules.vtk.serializers import configure_serializer

from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkPiecewiseFunction
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkIOImage import vtkTIFFReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

# Required for interactor initialization
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa

# Required for rendering initialization
import vtkmodules.vtkRenderingOpenGL2  # noqa


# -----------------------------------------------------------------------------
# Paths & configuration
# -----------------------------------------------------------------------------

CURRENT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = (CURRENT_DIRECTORY / ".." / "data").resolve()
if not DATA_DIRECTORY.exists():
    DATA_DIRECTORY = CURRENT_DIRECTORY
DEFAULT_TIFF = next((str(path) for path in DATA_DIRECTORY.glob("*.tif*")), "")

# Configure scene encoder
configure_serializer(encode_lut=True, skip_light=True)


# -----------------------------------------------------------------------------
# Constants & enums
# -----------------------------------------------------------------------------

class Representation(IntEnum):
    POINTS = 0
    WIREFRAME = 1
    SURFACE = 2
    SURFACE_WITH_EDGES = 3


class LookupTable(IntEnum):
    RAINBOW = 0
    INVERTED_RAINBOW = 1
    GREYSCALE = 2
    INVERTED_GREYSCALE = 3


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def coerce_float(value, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def valid_bounds(bounds) -> bool:
    return (
        bounds
        and bounds[1] > bounds[0]
        and bounds[3] > bounds[2]
        and bounds[5] > bounds[4]
    )


def build_dataset_arrays(reader: vtkXMLUnstructuredGridReader) -> list[dict]:
    """Extract array metadata from point and cell data."""
    result = []
    output = reader.GetOutput()

    fields = [
        (output.GetPointData(), vtkDataObject.FIELD_ASSOCIATION_POINTS),
        (output.GetCellData(), vtkDataObject.FIELD_ASSOCIATION_CELLS),
    ]

    for field_arrays, association in fields:
        for i in range(field_arrays.GetNumberOfArrays()):
            array = field_arrays.GetArray(i)
            if array is None:
                continue
            array_range = array.GetRange()
            result.append(
                {
                    "text": array.GetName(),
                    "value": i,
                    "range": list(array_range),
                    "type": association,
                }
            )
    return result


def apply_lut_preset(lut, preset: LookupTable) -> None:
    """Configure LUT according to a preset."""
    if preset == LookupTable.RAINBOW:
        lut.SetHueRange(0.666, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif preset == LookupTable.INVERTED_RAINBOW:
        lut.SetHueRange(0.0, 0.666)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif preset == LookupTable.GREYSCALE:
        lut.SetHueRange(0.0, 0.0)
        lut.SetSaturationRange(0.0, 0.0)
        lut.SetValueRange(0.0, 1.0)
    elif preset == LookupTable.INVERTED_GREYSCALE:
        lut.SetHueRange(0.0, 0.666)
        lut.SetSaturationRange(0.0, 0.0)
        lut.SetValueRange(1.0, 0.0)
    lut.Build()


def update_representation(actor: vtkActor, mode: Representation) -> None:
    prop = actor.GetProperty()
    if mode == Representation.POINTS:
        prop.SetRepresentationToPoints()
        prop.SetPointSize(5)
        prop.EdgeVisibilityOff()
    elif mode == Representation.WIREFRAME:
        prop.SetRepresentationToWireframe()
        prop.SetPointSize(1)
        prop.EdgeVisibilityOff()
    elif mode == Representation.SURFACE:
        prop.SetRepresentationToSurface()
        prop.SetPointSize(1)
        prop.EdgeVisibilityOff()
    elif mode == Representation.SURFACE_WITH_EDGES:
        prop.SetRepresentationToSurface()
        prop.SetPointSize(1)
        prop.EdgeVisibilityOn()


def color_by_array(actor: vtkActor, array_info: dict) -> None:
    """Apply scalar coloring to an actor using array metadata."""
    mapper: vtkDataSetMapper = actor.GetMapper()
    _min, _max = array_info["range"]

    mapper.SelectColorArray(array_info["text"])
    lut = mapper.GetLookupTable()
    lut.SetRange(_min, _max)

    if array_info["type"] == vtkDataObject.FIELD_ASSOCIATION_POINTS:
        mapper.SetScalarModeToUsePointFieldData()
    else:
        mapper.SetScalarModeToUseCellFieldData()

    mapper.SetScalarVisibility(True)
    mapper.SetUseLookupTableScalarRange(True)


def describe_unit_distance(image_data) -> float:
    spacing = image_data.GetSpacing()
    if not spacing:
        return 1.0
    return max(float(max(spacing)), 1e-3)


# -----------------------------------------------------------------------------
# VTK pipeline
# -----------------------------------------------------------------------------

renderer = vtkRenderer()
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)
render_window_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

# Read Data
reader = vtkXMLUnstructuredGridReader()
reader.SetFileName(str((CURRENT_DIRECTORY / ".." / "data" / "disk_out_ref.vtu").resolve()))
reader.Update()

# Dataset arrays
dataset_arrays = build_dataset_arrays(reader)
default_array = dataset_arrays[0]
default_min, default_max = default_array["range"]

# Mesh
mesh_mapper = vtkDataSetMapper()
mesh_mapper.SetInputConnection(reader.GetOutputPort())
mesh_actor = vtkActor()
mesh_actor.SetMapper(mesh_mapper)
renderer.AddActor(mesh_actor)

# Mesh: default representation (surface)
update_representation(mesh_actor, Representation.SURFACE)
mesh_actor.GetProperty().SetPointSize(1)
mesh_actor.GetProperty().EdgeVisibilityOff()

# Mesh: LUT & Color
mesh_lut = mesh_mapper.GetLookupTable()
apply_lut_preset(mesh_lut, LookupTable.RAINBOW)
color_by_array(mesh_actor, default_array)

# Contour
contour = vtkContourFilter()
contour.SetInputConnection(reader.GetOutputPort())
contour_mapper = vtkDataSetMapper()
contour_mapper.SetInputConnection(contour.GetOutputPort())
contour_actor = vtkActor()
contour_actor.SetMapper(contour_mapper)
renderer.AddActor(contour_actor)

# Contour: default array/values
contour_value = 0.5 * (default_max + default_min)
contour.SetInputArrayToProcess(
    0, 0, 0, default_array["type"], default_array["text"]
)
contour.SetValue(0, contour_value)

# Contour: representation & LUT
update_representation(contour_actor, Representation.SURFACE)
contour_actor.GetProperty().SetPointSize(1)
contour_actor.GetProperty().EdgeVisibilityOff()

contour_lut = contour_mapper.GetLookupTable()
apply_lut_preset(contour_lut, LookupTable.RAINBOW)
color_by_array(contour_actor, default_array)

# Volume (TIFF or synthetic)
volume_mapper = vtkSmartVolumeMapper()
volume_mapper.SetBlendModeToComposite()

volume_color_function = vtkColorTransferFunction()
volume_opacity_function = vtkPiecewiseFunction()

volume_property = vtkVolumeProperty()
volume_property.SetIndependentComponents(True)
volume_property.SetInterpolationTypeToLinear()
volume_property.SetColor(volume_color_function)
volume_property.SetScalarOpacity(volume_opacity_function)
volume_property.ShadeOn()

volume_actor = vtkVolume()
volume_actor.SetMapper(volume_mapper)
volume_actor.SetProperty(volume_property)
volume_actor.VisibilityOff()
renderer.AddVolume(volume_actor)

# Cube Axes
cube_axes = vtkCubeAxesActor()
renderer.AddActor(cube_axes)
cube_axes.SetBounds(mesh_actor.GetBounds())
cube_axes.SetCamera(renderer.GetActiveCamera())
cube_axes.SetXLabelFormat("%6.1f")
cube_axes.SetYLabelFormat("%6.1f")
cube_axes.SetZLabelFormat("%6.1f")
cube_axes.SetFlyModeToOuterEdges()

renderer.ResetCamera()

# Volume state
_volume_reader = None
_volume_opacity_nodes: list[tuple[float, float, float, float]] = []


def trigger_view_update():
    controller = globals().get("ctrl")
    if controller is None:
        return
    update_fn = getattr(controller, "view_update", None)
    if callable(update_fn):
        update_fn()


def update_scene_bounds():
    combined = None
    for vtk_object in (mesh_actor, contour_actor, volume_actor):
        if vtk_object is None:
            continue
        if hasattr(vtk_object, "GetVisibility") and not vtk_object.GetVisibility():
            continue
        bounds = vtk_object.GetBounds()
        if not valid_bounds(bounds):
            continue
        if combined is None:
            combined = list(bounds)
        else:
            combined[0] = min(combined[0], bounds[0])
            combined[1] = max(combined[1], bounds[1])
            combined[2] = min(combined[2], bounds[2])
            combined[3] = max(combined[3], bounds[3])
            combined[4] = min(combined[4], bounds[4])
            combined[5] = max(combined[5], bounds[5])

    if combined:
        cube_axes.SetBounds(combined)
        renderer.ResetCameraClippingRange()


def rebuild_volume_opacity_baseline():
    _volume_opacity_nodes.clear()
    buffer = [0.0, 0.0, 0.0, 0.0]
    for idx in range(volume_opacity_function.GetSize()):
        volume_opacity_function.GetNodeValue(idx, buffer)
        _volume_opacity_nodes.append(tuple(buffer))


def apply_volume_opacity_factor(factor: float):
    if not _volume_opacity_nodes:
        return
    factor = max(0.0, factor)
    adjusted = vtkPiecewiseFunction()
    for x, y, midpoint, sharpness in _volume_opacity_nodes:
        adjusted.AddPoint(x, max(0.0, min(y * factor, 1.0)), midpoint, sharpness)
    volume_property.SetScalarOpacity(adjusted)


def configure_volume_transfer_functions(data_range):
    minimum, maximum = data_range
    if maximum <= minimum:
        maximum = minimum + 1.0

    span = maximum - minimum
    midpoint = minimum + 0.5 * span

    volume_color_function.RemoveAllPoints()
    volume_color_function.AddRGBPoint(minimum, 0.1, 0.1, 0.25)
    volume_color_function.AddRGBPoint(midpoint, 0.4, 0.7, 0.9)
    volume_color_function.AddRGBPoint(maximum, 0.95, 0.95, 0.95)

    volume_opacity_function.RemoveAllPoints()
    ramp_points = [
        (minimum, 0.0),
        (minimum + 0.1 * span, 0.02),
        (minimum + 0.45 * span, 0.12),
        (minimum + 0.75 * span, 0.35),
        (maximum, 0.8),
    ]
    for x, y in ramp_points:
        volume_opacity_function.AddPoint(x, y, 0.5, 0.0)

    rebuild_volume_opacity_baseline()


def load_volume_from_path(volume_path: str):
    global _volume_reader

    state_obj = globals().get("state")
    if state_obj is None:
        return

    status_message = ""
    resolved_path = Path(volume_path).expanduser() if volume_path else None

    try:
        if resolved_path and resolved_path.is_file():
            reader = vtkTIFFReader()
            reader.SetFileDimensionality(3)
            reader.SetFileName(str(resolved_path))
            reader.Update()
            _volume_reader = reader

            image_data = reader.GetOutput()
            volume_mapper.SetInputConnection(reader.GetOutputPort())
            data_range = image_data.GetScalarRange()
            configure_volume_transfer_functions(data_range)
            apply_volume_opacity_factor(
                coerce_float(getattr(state_obj, "volume_opacity_factor", 1.0))
            )
            volume_property.SetShade(bool(getattr(state_obj, "volume_shade", True)))
            volume_property.SetScalarOpacityUnitDistance(describe_unit_distance(image_data))

            dims = image_data.GetDimensions()
            state_obj.volume_data_range = f"{data_range[0]:.3f} – {data_range[1]:.3f}"
            state_obj.volume_dimensions = " × ".join(str(v) for v in dims)
            state_obj.volume_visible = True
            volume_actor.VisibilityOn()
        else:
            if resolved_path:
                status_message = f"File not found: {resolved_path}"
            volume_mapper.RemoveAllInputs()
            volume_actor.VisibilityOff()
            state_obj.volume_visible = False
            state_obj.volume_data_range = "—"
            state_obj.volume_dimensions = "—"
    except Exception as error:  # noqa: BLE001
        status_message = f"Failed to load volume: {error}"
        volume_mapper.RemoveAllInputs()
        volume_actor.VisibilityOff()
        state_obj.volume_visible = False
    finally:
        state_obj.volume_status = status_message
        update_scene_bounds()
        renderer.ResetCamera()
        trigger_view_update()


# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

state.setdefault("active_ui", "mesh")
state.setdefault("mesh_visible", True)
state.setdefault("contour_visible", True)
state.setdefault("volume_visible", False)
state.setdefault("volume_tiff_path", DEFAULT_TIFF)
state.setdefault("volume_status", "")
state.setdefault("volume_opacity_factor", 1.0)
state.setdefault("volume_shade", True)
state.setdefault("volume_data_range", "—")
state.setdefault("volume_dimensions", "—")


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@state.change("cube_axes_visibility")
def update_cube_axes_visibility(cube_axes_visibility, **kwargs):
    cube_axes.SetVisibility(cube_axes_visibility)
    trigger_view_update()


@state.change("mesh_visible")
def update_mesh_visibility(mesh_visible, **kwargs):
    mesh_actor.SetVisibility(bool(mesh_visible))
    update_scene_bounds()
    trigger_view_update()


@state.change("contour_visible")
def update_contour_visibility(contour_visible, **kwargs):
    contour_actor.SetVisibility(bool(contour_visible))
    update_scene_bounds()
    trigger_view_update()


@state.change("mesh_representation")
def update_mesh_representation(mesh_representation, **kwargs):
    update_representation(mesh_actor, Representation(mesh_representation))
    trigger_view_update()


@state.change("contour_representation")
def update_contour_representation(contour_representation, **kwargs):
    update_representation(contour_actor, Representation(contour_representation))
    trigger_view_update()


@state.change("mesh_color_array_idx")
def update_mesh_color_by_name(mesh_color_array_idx, **kwargs):
    array = dataset_arrays[mesh_color_array_idx]
    color_by_array(mesh_actor, array)
    trigger_view_update()


@state.change("contour_color_array_idx")
def update_contour_color_by_name(contour_color_array_idx, **kwargs):
    array = dataset_arrays[contour_color_array_idx]
    color_by_array(contour_actor, array)
    trigger_view_update()


@state.change("mesh_color_preset")
def update_mesh_color_preset(mesh_color_preset, **kwargs):
    lut = mesh_actor.GetMapper().GetLookupTable()
    apply_lut_preset(lut, LookupTable(mesh_color_preset))
    trigger_view_update()


@state.change("contour_color_preset")
def update_contour_color_preset(contour_color_preset, **kwargs):
    lut = contour_actor.GetMapper().GetLookupTable()
    apply_lut_preset(lut, LookupTable(contour_color_preset))
    trigger_view_update()


@state.change("mesh_opacity")
def update_mesh_opacity(mesh_opacity, **kwargs):
    mesh_actor.GetProperty().SetOpacity(mesh_opacity)
    trigger_view_update()


@state.change("contour_opacity")
def update_contour_opacity(contour_opacity, **kwargs):
    contour_actor.GetProperty().SetOpacity(contour_opacity)
    trigger_view_update()


@state.change("contour_by_array_idx")
def update_contour_by(contour_by_array_idx, **kwargs):
    array = dataset_arrays[contour_by_array_idx]
    contour_min, contour_max = array["range"]
    contour_step = 0.01 * (contour_max - contour_min)
    contour_value_local = 0.5 * (contour_max + contour_min)

    contour.SetInputArrayToProcess(0, 0, 0, array["type"], array["text"])
    contour.SetValue(0, contour_value_local)

    # Update UI
    state.contour_min = contour_min
    state.contour_max = contour_max
    state.contour_value = contour_value_local
    state.contour_step = contour_step

    trigger_view_update()


@state.change("contour_value")
def update_contour_value(contour_value, **kwargs):
    contour.SetValue(0, float(contour_value))
    trigger_view_update()


@state.change("volume_visible")
def update_volume_visibility(volume_visible, **kwargs):
    volume_actor.SetVisibility(bool(volume_visible))
    update_scene_bounds()
    trigger_view_update()


@state.change("volume_opacity_factor")
def update_volume_opacity(volume_opacity_factor, **kwargs):
    apply_volume_opacity_factor(coerce_float(volume_opacity_factor))
    trigger_view_update()


@state.change("volume_shade")
def update_volume_shading(volume_shade, **kwargs):
    volume_property.SetShade(bool(volume_shade))
    trigger_view_update()


def reload_volume():
    load_volume_from_path(state.volume_tiff_path or "")


ctrl.volume_reload = reload_volume


# -----------------------------------------------------------------------------
# GUI elements
# -----------------------------------------------------------------------------

def standard_buttons():
    vuetify.VCheckbox(
        v_model=("cube_axes_visibility", True),
        on_icon="mdi-cube-outline",
        off_icon="mdi-cube-off-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model="$vuetify.theme.dark",
        on_icon="mdi-lightbulb-off-outline",
        off_icon="mdi-lightbulb-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model=("viewMode", "local"),
        on_icon="mdi-lan-disconnect",
        off_icon="mdi-lan-connect",
        true_value="local",
        false_value="remote",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VBtn(icon=True, click="$refs.view.resetCamera()"):
        vuetify.VIcon("mdi-crop-free")


def pipeline_widget():
    with vuetify.VCard(classes="mb-2"):
        vuetify.VCardTitle(
            "Pipeline",
            classes="grey lighten-1 py-1 grey--text text--darken-3",
            style="user-select: none; cursor: pointer",
        )
        with vuetify.VCardText(classes="py-0"):
            with vuetify.VList(dense=True, nav=True):
                with vuetify.VListItemGroup(
                    v_model=("active_ui", "mesh"), mandatory=True
                ):
                    for value, label, state_key in [
                        ("mesh", "Mesh", "mesh_visible"),
                        ("contour", "Contour", "contour_visible"),
                        ("volume", "Volume", "volume_visible"),
                    ]:
                        with vuetify.VListItem(value=value, dense=True, ripple=False):
                            with vuetify.VListItemContent():
                                vuetify.VListItemTitle(label)
                            with vuetify.VListItemAction() as action:
                                action.ripple = False
                                vuetify.VCheckbox(
                                    v_model=(state_key, True),
                                    hide_details=True,
                                    dense=True,
                                    on_icon="mdi-eye",
                                    off_icon="mdi-eye-off",
                                    true_value=True,
                                    false_value=False,
                                    classes="ma-0 pa-0",
                                )


def ui_card(title: str, ui_name: str):
    with vuetify.VCard(v_show=f"active_ui == '{ui_name}'"):
        vuetify.VCardTitle(
            title,
            classes="grey lighten-1 py-1 grey--text text--darken-3",
            style="user-select: none; cursor: pointer",
            hide_details=True,
            dense=True,
        )
        content = vuetify.VCardText(classes="py-2")
    return content


def mesh_card():
    with ui_card(title="Mesh", ui_name="mesh"):
        vuetify.VSelect(
            v_model=("mesh_representation", Representation.SURFACE),
            items=(
                "representations",
                [
                    {"text": "Points", "value": int(Representation.POINTS)},
                    {"text": "Wireframe", "value": int(Representation.WIREFRAME)},
                    {"text": "Surface", "value": int(Representation.SURFACE)},
                    {
                        "text": "SurfaceWithEdges",
                        "value": int(Representation.SURFACE_WITH_EDGES),
                    },
                ],
            ),
            label="Representation",
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1",
        )
        with vuetify.VRow(classes="pt-2", dense=True):
            with vuetify.VCol(cols="6"):
                vuetify.VSelect(
                    label="Color by",
                    v_model=("mesh_color_array_idx", 0),
                    items=("array_list", dataset_arrays),
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
            with vuetify.VCol(cols="6"):
                vuetify.VSelect(
                    label="Colormap",
                    v_model=("mesh_color_preset", LookupTable.RAINBOW),
                    items=(
                        "colormaps",
                        [
                            {"text": "Rainbow", "value": int(LookupTable.RAINBOW)},
                            {
                                "text": "Inv Rainbow",
                                "value": int(LookupTable.INVERTED_RAINBOW),
                            },
                            {"text": "Greyscale", "value": int(LookupTable.GREYSCALE)},
                            {
                                "text": "Inv Greyscale",
                                "value": int(LookupTable.INVERTED_GREYSCALE),
                            },
                        ],
                    ),
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
        vuetify.VSlider(
            v_model=("mesh_opacity", 1.0),
            min=0,
            max=1,
            step=0.1,
            label="Opacity",
            classes="mt-1",
            hide_details=True,
            dense=True,
        )


def contour_card():
    with ui_card(title="Contour", ui_name="contour"):
        vuetify.VSelect(
            label="Contour by",
            v_model=("contour_by_array_idx", 0),
            items=("array_list", dataset_arrays),
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1",
        )
        vuetify.VSlider(
            v_model=("contour_value", contour_value),
            min=("contour_min", default_min),
            max=("contour_max", default_max),
            step=("contour_step", 0.01 * (default_max - default_min)),
            label="Value",
            classes="my-1",
            hide_details=True,
            dense=True,
        )
        vuetify.VSelect(
            v_model=("contour_representation", Representation.SURFACE),
            items=(
                "representations",
                [
                    {"text": "Points", "value": int(Representation.POINTS)},
                    {"text": "Wireframe", "value": int(Representation.WIREFRAME)},
                    {"text": "Surface", "value": int(Representation.SURFACE)},
                    {
                        "text": "SurfaceWithEdges",
                        "value": int(Representation.SURFACE_WITH_EDGES),
                    },
                ],
            ),
            label="Representation",
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1",
        )
        with vuetify.VRow(classes="pt-2", dense=True):
            with vuetify.VCol(cols="6"):
                vuetify.VSelect(
                    label="Color by",
                    v_model=("contour_color_array_idx", 0),
                    items=("array_list", dataset_arrays),
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
            with vuetify.VCol(cols="6"):
                vuetify.VSelect(
                    label="Colormap",
                    v_model=("contour_color_preset", LookupTable.RAINBOW),
                    items=(
                        "colormaps",
                        [
                            {"text": "Rainbow", "value": int(LookupTable.RAINBOW)},
                            {
                                "text": "Inv Rainbow",
                                "value": int(LookupTable.INVERTED_RAINBOW),
                            },
                            {"text": "Greyscale", "value": int(LookupTable.GREYSCALE)},
                            {
                                "text": "Inv Greyscale",
                                "value": int(LookupTable.INVERTED_GREYSCALE),
                            },
                        ],
                    ),
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
        vuetify.VSlider(
            v_model=("contour_opacity", 1.0),
            min=0,
            max=1,
            step=0.1,
            label="Opacity",
            classes="mt-1",
            hide_details=True,
            dense=True,
        )


def volume_card():
    with ui_card(title="Volume", ui_name="volume"):
        vuetify.VTextField(
            v_model=("volume_tiff_path", state.volume_tiff_path or ""),
            label="TIFF file path",
            placeholder="/path/to/volume.tif",
            clearable=True,
            hide_details=True,
            dense=True,
            classes="pt-1",
        )
        vuetify.VBtn(
            "Load Volume",
            color="primary",
            block=True,
            classes="mt-2",
            click=ctrl.volume_reload,
        )
        vuetify.VSlider(
            v_model=("volume_opacity_factor", 1.0),
            min=0.1,
            max=5.0,
            step=0.05,
            hide_details=True,
            dense=True,
            label="Opacity multiplier",
            classes="mt-4",
        )
        vuetify.VSwitch(
            v_model=("volume_shade", True),
            label="Enable shading",
            hide_details=True,
            dense=True,
            classes="mt-4",
        )
        with vuetify.VAlert(
            v_show="volume_status",
            type="error",
            dense=True,
            outlined=True,
            classes="mt-3",
        ):
            html.Div("{{ volume_status }}")
        html.Div("Scalar range: {{ volume_data_range }}", classes="text-caption mt-3")
        html.Div("Dimensions: {{ volume_dimensions }}", classes="text-caption")


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Viewer")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VDivider(vertical=True, classes="mx-2")
        standard_buttons()

    with layout.drawer as drawer:
        drawer.width = 325
        pipeline_widget()
        vuetify.VDivider(classes="mb-2")
        mesh_card()
        contour_card()
        volume_card()

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            view = vtk.VtkRemoteLocalView(
                render_window, namespace="view", mode="local", interactive_ratio=1
            )
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

            load_volume_from_path(state.volume_tiff_path or "")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
