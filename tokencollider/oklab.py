"""Map layout dims 4-6 to a perceptually uniform color (Oklab -> sRGB).

Equal distances along the color axes should read as equal color differences,
so the viewport's color channel lies no more than its spatial channels do.
Reference: Björn Ottosson, https://bottosson.github.io/posts/oklab/
"""

import numpy as np

# z-scores (coord / axis std) land here: L is lightness, a/b are chroma axes.
L_CENTER, L_SPAN = 0.72, 0.10
AB_SPAN = 0.12
Z_CLAMP = 2.0


def oklab_to_srgb(L: float, a: float, b: float) -> tuple[int, int, int]:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    lin = np.array([
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ])
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return tuple(int(round(c * 255)) for c in srgb)


def srgb_to_oklab(r: int, g: int, b: int) -> tuple[float, float, float]:
    srgb = np.array([r, g, b], dtype=np.float64) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    l = 0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2]
    m = 0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2]
    s = 0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2]
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def zscores_to_hex(z4: float, z5: float, z6: float) -> str:
    """Layout color-axis z-scores -> '#rrggbb'."""
    z4, z5, z6 = (float(np.clip(z, -Z_CLAMP, Z_CLAMP)) for z in (z4, z5, z6))
    r, g, b = oklab_to_srgb(
        L_CENTER + L_SPAN * (z4 / Z_CLAMP),
        AB_SPAN * (z5 / Z_CLAMP),
        AB_SPAN * (z6 / Z_CLAMP),
    )
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_zscores(color: str) -> tuple[float, float, float]:
    """Inverse of zscores_to_hex (within gamut): '#rrggbb' -> z-scores."""
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    L, a, bb = srgb_to_oklab(r, g, b)
    return (
        (L - L_CENTER) / L_SPAN * Z_CLAMP,
        a / AB_SPAN * Z_CLAMP,
        bb / AB_SPAN * Z_CLAMP,
    )
