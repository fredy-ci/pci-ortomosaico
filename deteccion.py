#deteccion y cuantificacion de fallas en una UM
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.morphology import skeletonize

from .config import CATALOGO_ASTM, Config, NOMBRES_FALLA, UNIDAD_MEDIDA

FACTOR_LONGITUD = 1.08


#utilidades
def _elipse(radio_px: float):
    k = max(1, int(round(radio_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _fondo_local(gray: np.ndarray, util: np.ndarray, sigma_px: float) -> np.ndarray:
    f = max(1, int(sigma_px // 6))
    g = gray if f == 1 else cv2.resize(gray, None, fx=1.0 / f, fy=1.0 / f, interpolation=cv2.INTER_AREA)
    u = util.astype(np.float32)
    u = u if f == 1 else cv2.resize(u, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
    s = sigma_px / f
    num = cv2.GaussianBlur(g * u, (0, 0), s)
    den = cv2.GaussianBlur(u, (0, 0), s)
    bg = num / np.maximum(den, 1e-3)
    medio = float(gray[util].mean()) if util.any() else 128.0
    bg = np.where(den > 1e-2, bg, medio)
    if f != 1:
        bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return bg.astype(np.float32)


def _plano(gs: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mascara)
    if len(xs) < 50:
        return np.full_like(gs, float(np.median(gs)))
    paso = max(1, len(xs) // 30000)
    xs, ys = xs[::paso], ys[::paso]
    z = gs[ys, xs]
    A = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    keep = np.ones(len(z), bool)
    coef = np.zeros(3)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        s = res[keep].std() + 1e-6
        keep = np.abs(res) < 2.5 * s
        if keep.sum() < 50:
            break
    H, W = gs.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    return (coef[0] * X + coef[1] * Y + coef[2]).astype(np.float32)


def _ancho_equivalente_mm(norm_g, crack, sk, gsd) -> float:
    n_sk = int(sk.sum())
    if n_sk == 0:
        return 0.0
    banda = cv2.dilate(crack.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    pico = max(1.0, float(np.percentile(-norm_g[sk], 75)))
    ancho_px = max(0.3, float((-norm_g[banda]).sum()) / (pico * n_sk))
    return ancho_px * gsd * 1000.0


def _sev_ancho(ancho_mm: float, cfg: Config) -> str:
    b, m = cfg.grieta_ancho_sev_mm
    return "B" if ancho_mm < b else ("M" if ancho_mm <= m else "A")


def _podar_esqueleto(sk: np.ndarray, k: int) -> np.ndarray:
    """quita las espinas cortas (<= k px) del esqueleto, sin acortar los extremos de las ramas principales."""
    sk0 = sk.astype(np.uint8)
    ker = np.ones((3, 3), np.float32)
    ker[1, 1] = 0
    def vecinos(a):
        return cv2.filter2D(a, cv2.CV_8U, ker, borderType=cv2.BORDER_CONSTANT)
    s = sk0.copy()
    for _ in range(k):
        fin = (s == 1) & (vecinos(s) <= 1)
        if not fin.any():
            break
        s[fin] = 0
    if not s.any():
        return sk.astype(bool)
    crece = ((s == 1) & (vecinos(s) <= 1)).astype(np.uint8)
    for _ in range(k):
        crece = cv2.dilate(crece, np.ones((3, 3), np.uint8)) & sk0 & (1 - s)
        s |= crece
    return s.astype(bool)


def _sev_bache(diam_mm: float, prof_mm: float) -> str:
    if prof_mm <= 25.4:
        col = 0
    elif prof_mm <= 50.8:
        col = 1
    else:
        col = 2
    if diam_mm < 200:
        fila = 0
    elif diam_mm < 450:
        fila = 1
    else:
        fila = 2
    tabla = [["B", "B", "M"],
             ["B", "M", "M"],
             ["M", "A", "A"]]
    return tabla[fila][col]


#deteccion por unidad
def detectar_unidad(img_bgr: np.ndarray, mascara: np.ndarray, gsd: float, cfg: Config,
                    offset: Tuple[int, int] = (0, 0), generar_overlay: bool = True, parches_usuario=None, mascara_calzada=None, lineas_auto=None, piel_forma=None, desprend_forma=None, bache_forma=None, bloque_forma=None, ahuell_forma=None, grieta_forma=None, dsm=None, zonas_extra=None) -> Dict:
    h, w = mascara.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    #mascara util 
    pintura_blanca = (hsv[:, :, 2] > cfg.marca_v_min) & (hsv[:, :, 1] < cfg.marca_s_max)
    pintura_amarilla = (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 110)
    marcas = (pintura_blanca | pintura_amarilla) & mascara
    marcas = cv2.dilate(marcas.astype(np.uint8), _elipse(0.03 / gsd)).astype(bool)
    if cfg.borde_excluir_m > 0:
        borde = cv2.erode(mascara.astype(np.uint8), _elipse(cfg.borde_excluir_m / gsd)).astype(bool)
    else:
        borde = mascara.astype(bool)
    util = borde & ~marcas

    parche_usuario = np.zeros((h, w), bool)
    ins_usuario: List[Dict] = []
    for pm in (parches_usuario or []):
        yy, xx = np.nonzero(pm)
        if len(yy) == 0:
            continue
        parche_usuario |= pm
        ins_usuario.append({"codigo": 11, "severidad": cfg.parche_usuario_sev, "valor": float(pm.sum()) * gsd * gsd,
                            "unidad_medida": "m2", "x": int(xx.min()) + offset[0], "y": int(yy.min()) + offset[1],
                            "w": int(xx.max() - xx.min() + 1), "h": int(yy.max() - yy.min() + 1),
                            "detalle": "parche dibujado por el usuario"})
    if parche_usuario.any():
        util &= ~(cv2.dilate(parche_usuario.astype(np.uint8), _elipse(0.03 / gsd)) > 0)

    if util.sum() < 100 and not ins_usuario:
        return {"fallas": [], "instancias": [], "overlay": None, "util_m2": 0.0}
    if util.sum() < 100:
        util[:] = False
        fallas_u: Dict[Tuple[int, str], float] = {}
        for ins in ins_usuario:
            fallas_u[(11, ins["severidad"])] = fallas_u.get((11, ins["severidad"]), 0.0) + ins["valor"]
        return {"fallas": [{"codigo": c, "nombre": NOMBRES_FALLA[c], "severidad": s, "cantidad": float(v),
                            "unidad_medida": UNIDAD_MEDIDA[c]} for (c, s), v in sorted(fallas_u.items())],
                "instancias": ins_usuario, "overlay": None, "util_m2": 0.0}

    fondo = _fondo_local(gray, util, sigma_px=0.5 / gsd)
    norm = np.where(util, gray - fondo, 0.0).astype(np.float32)

    #baches
    bache_mask = np.zeros((h, w), bool)
    instancias: List[Dict] = list(ins_usuario)
    s_b = cv2.GaussianBlur(norm, (0, 0), max(1.0, 0.02 / gsd))
    cand = ((s_b < -cfg.bache_contraste) & util).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _elipse(0.03 / gsd))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _elipse(0.02 / gsd))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    baches = []
    for i in range(1, n):
        a_px = st[i, cv2.CC_STAT_AREA]
        a_m2 = a_px * gsd * gsd
        diam_mm = math.sqrt(4 * a_m2 / math.pi) * 1000
        if diam_mm < cfg.bache_diam_min_mm or a_m2 > 6.0:
            continue
        x, y, bw, bh = st[i, :4]
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnt, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnt:
            continue
        c = max(cnt, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * cv2.contourArea(c) / (per * per + 1e-9)
        hull = cv2.contourArea(cv2.convexHull(c))
        solid = cv2.contourArea(c) / (hull + 1e-9)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        aspecto = max(rw, rh) / (min(rw, rh) + 1e-9)
        if circ < cfg.bache_circularidad_min or solid < 0.75 or aspecto > 3.5:
            continue
        pad = int(0.10 / gsd)
        ya, yb = max(0, y - pad), min(h, y + bh + pad)
        xa, xb = max(0, x - pad), min(w, x + bw + pad)
        sb = s_b[ya:yb, xa:xb]
        comp_full = lab[ya:yb, xa:xb] == i
        prof = -float(sb[comp_full].min())
        reg = ((sb < -0.5 * prof) & util[ya:yb, xa:xb]).astype(np.uint8)
        nr, lr = cv2.connectedComponents(reg, connectivity=8)
        ids = np.unique(lr[comp_full & (lr > 0)])
        mk = np.isin(lr, ids[ids > 0])
        if mk.sum() < comp_full.sum():
            mk = comp_full
        a_m2 = float(mk.sum()) * gsd * gsd
        diam_mm = math.sqrt(4 * a_m2 / math.pi) * 1000
        yy, xx = np.nonzero(mk)
        baches.append((i, xa + int(xx.min()), ya + int(yy.min()), int(xx.max() - xx.min() + 1),
                       int(yy.max() - yy.min() + 1), a_m2, diam_mm, (ya, yb, xa, xb, mk)))
    baches_sev: List[str] = []
    for (i, x, y, bw, bh, a_m2, diam_mm, (ya, yb, xa, xb, mk)) in baches:
        bache_mask[ya:yb, xa:xb] |= mk
        if diam_mm > 750:
            cant = max(1.0, round(a_m2 / 0.5))
            sev = "M" if cfg.bache_profundidad_mm <= 25.0 else "A"
        else:
            cant = 1.0
            sev = _sev_bache(diam_mm, cfg.bache_profundidad_mm)
        baches_sev.append(sev)
        instancias.append({"codigo": 13, "severidad": sev, "valor": cant, "unidad_medida": "und",
                           "x": x + offset[0], "y": y + offset[1], "w": bw, "h": bh,
                           "detalle": f"diametro_eq={diam_mm:.0f}mm; prof_supuesta={cfg.bache_profundidad_mm:.0f}mm"})
    for bf in (bache_forma or []):
        mk_ = bf & util
        a_m2 = float(mk_.sum()) * gsd * gsd
        if a_m2 < 0.03 or (mk_ & bache_mask).sum() > 0.5 * mk_.sum():
            continue
        diam_mm = math.sqrt(4 * a_m2 / math.pi) * 1000
        yy, xx = np.nonzero(mk_)
        if diam_mm > 750:
            cant = max(1.0, round(a_m2 / 0.5))
            sev = "M" if cfg.bache_profundidad_mm <= 25.0 else "A"
        else:
            cant = 1.0
            sev = _sev_bache(diam_mm, cfg.bache_profundidad_mm)
        bache_mask |= mk_
        x_, y_ = int(xx.min()), int(yy.min())
        bw_, bh_ = int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)
        baches.append((-1, x_, y_, bw_, bh_, a_m2, diam_mm, (0, h, 0, w, mk_)))
        baches_sev.append(sev)
        instancias.append({"codigo": 13, "severidad": sev, "valor": cant, "unidad_medida": "und",
                           "x": x_ + offset[0], "y": y_ + offset[1], "w": bw_, "h": bh_,
                           "detalle": f"plantilla; diametro_eq={diam_mm:.0f}mm; prof_supuesta={cfg.bache_profundidad_mm:.0f}mm"})
    bache_dil = cv2.dilate(bache_mask.astype(np.uint8), _elipse(0.04 / gsd)).astype(bool)

    #grietas (candidatas)
    sig = [max(1.0, mm / 1000.0 / gsd) for mm in cfg.grieta_sigmas_mm]
    norm_g = np.where(bache_dil, 0.0, norm)
    resp = sato(norm_g, sigmas=sig, black_ridges=True, mode="reflect").astype(np.float32)
    resp[~util] = 0
    vals = resp[util]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-6
    z = (resp - med) / mad
    cr = apply_hysteresis_threshold(z, cfg.grieta_z_bajo, cfg.grieta_z_alto)
    oscuro = cv2.GaussianBlur(norm_g, (0, 0), 0.8) < -cfg.grieta_contraste_min
    oscuro = cv2.dilate(oscuro.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    mancha_dil = np.zeros((h, w), bool)
    if cfg.mancha_contraste > 0:
        mn = (cv2.GaussianBlur(norm_g, (0, 0), 0.8) < -cfg.mancha_contraste) & util
        if mn.any():
            mn = cv2.dilate(mn.astype(np.uint8), _elipse(cfg.mancha_margen_m / gsd))
            mn = cv2.morphologyEx(mn, cv2.MORPH_CLOSE, _elipse(cfg.mancha_union_m / gsd))
            nm_, lm_, sm_, _ = cv2.connectedComponentsWithStats(mn, connectivity=8)
            for i_ in range(1, nm_):
                if sm_[i_, cv2.CC_STAT_AREA] * gsd * gsd >= cfg.mancha_area_min_m2:
                    mancha_dil |= lm_ == i_
    crack_raw = (cr & oscuro & util & ~bache_dil & ~mancha_dil)
    crack_raw = cv2.morphologyEx(crack_raw.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)

    sk_raw = skeletonize(crack_raw)
    if cfg.recta_rms_max > 0 and sk_raw.any():
        sk8 = sk_raw.astype(np.uint8)
        nb_ = cv2.filter2D(sk8, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk8
        cruces_ = ((sk8 > 0) & (nb_ >= 3)).astype(np.uint8)
        tramos_ = sk8 & ~cv2.dilate(cruces_, np.ones((3, 3), np.uint8))
        nt_, lt_, st_, _ = cv2.connectedComponentsWithStats(tramos_, connectivity=8)
        rectas_ = np.zeros((h, w), np.uint8)
        for e in range(1, nt_):
            if st_[e, cv2.CC_STAT_AREA] < 10:
                continue
            yy_, xx_ = np.nonzero(lt_[st_[e, 1]:st_[e, 1] + st_[e, 3], st_[e, 0]:st_[e, 0] + st_[e, 2]] == e)
            pts_ = np.column_stack([xx_ + st_[e, 0], yy_ + st_[e, 1]]).astype(np.float32)
            vx, vy, x0_, y0_ = cv2.fitLine(pts_, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            t_ = (pts_[:, 0] - x0_) * vx + (pts_[:, 1] - y0_) * vy
            n_ = (pts_[:, 0] - x0_) * (-vy) + (pts_[:, 1] - y0_) * vx
            largo_ = float(t_.max() - t_.min()) * gsd
            if largo_ >= cfg.recta_largo_min_m and float(np.sqrt(np.mean(n_ ** 2))) * gsd / largo_ < cfg.recta_rms_max:
                rectas_[lt_ == e] = 1
        if rectas_.any():
            crack_raw &= ~(cv2.dilate(rectas_, _elipse(0.05 / gsd)) > 0)
        sk_raw = skeletonize(crack_raw)
    if lineas_auto is not None and cfg.cable_ang_tol > 0 and lineas_auto.any() and sk_raw.any():
        ys_c, xs_c = np.nonzero(skeletonize(lineas_auto))
        if len(xs_c) > 20:
            arbol = cKDTree(np.column_stack([xs_c, ys_c]))
            sk8 = sk_raw.astype(np.uint8)
            nb_ = cv2.filter2D(sk8, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk8
            tramos_ = sk8 & ~cv2.dilate(((sk8 > 0) & (nb_ >= 3)).astype(np.uint8), np.ones((3, 3), np.uint8))
            nt_, lt_, st_, _ = cv2.connectedComponentsWithStats(tramos_, connectivity=8)
            rectas_ = np.zeros((h, w), np.uint8)
            radio = cfg.cable_radio_m / gsd
            ventana = 0.8 / gsd
            for e in range(1, nt_):
                if st_[e, cv2.CC_STAT_AREA] < 8:
                    continue
                yy_, xx_ = np.nonzero(lt_[st_[e, 1]:st_[e, 1] + st_[e, 3], st_[e, 0]:st_[e, 0] + st_[e, 2]] == e)
                pts_ = np.column_stack([xx_ + st_[e, 0], yy_ + st_[e, 1]]).astype(np.float32)
                vx, vy, x0_, y0_ = cv2.fitLine(pts_, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                t_ = (pts_[:, 0] - x0_) * vx + (pts_[:, 1] - y0_) * vy
                n_ = (pts_[:, 0] - x0_) * (-vy) + (pts_[:, 1] - y0_) * vx
                largo_ = float(t_.max() - t_.min()) * gsd
                if largo_ < cfg.cable_largo_min_m or float(np.sqrt(np.mean(n_ ** 2))) * gsd / largo_ > cfg.cable_rms_max:
                    continue
                ang = math.degrees(math.atan2(vy, vx)) % 180.0
                if min(ang, 180.0 - ang) < cfg.cable_ang_min_eje:
                    continue
                dist_q, _ = arbol.query([float(x0_), float(y0_)])
                if dist_q > radio:
                    continue
                vec = arbol.query_ball_point([float(x0_), float(y0_)], radio)
                celdas_ = {}
                for j_ in vec:
                    celdas_.setdefault((int(xs_c[j_] // ventana), int(ys_c[j_] // ventana)), j_)
                coinc = False
                for j_ in celdas_.values():
                    cer_ = arbol.query_ball_point(arbol.data[j_], ventana)
                    if len(cer_) < 12:
                        continue
                    pc = np.column_stack([xs_c[cer_], ys_c[cer_]]).astype(np.float32)
                    ux, uy, _, _ = cv2.fitLine(pc, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                    ang_c = math.degrees(math.atan2(uy, ux)) % 180.0
                    if min(abs(ang - ang_c), 180.0 - abs(ang - ang_c)) <= cfg.cable_ang_tol:
                        coinc = True
                        break
                if coinc:
                    rectas_[lt_ == e] = 1
            if rectas_.any():
                crack_raw &= ~(cv2.dilate(rectas_, _elipse(0.05 / gsd)) > 0)
                sk_raw = skeletonize(crack_raw)
    if lineas_auto is not None and cfg.cable_cerca_m > 0 and lineas_auto.any() and crack_raw.any():
        dt_l = cv2.distanceTransform((~lineas_auto.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
        cerca_ = dt_l <= cfg.cable_cerca_m / gsd
        nl_, ll_, st__, _ = cv2.connectedComponentsWithStats(crack_raw.astype(np.uint8), connectivity=8)
        perp_ = st__[:, cv2.CC_STAT_HEIGHT] > 1.5 * st__[:, cv2.CC_STAT_WIDTH]   #cruza la calzada no es grieta de borde
        sel_ = ll_ > 0
        tot_ = np.bincount(ll_[sel_], minlength=nl_).astype(np.float64)
        cer_ = np.bincount(ll_[sel_], weights=cerca_[sel_].astype(np.float64), minlength=nl_)
        mcal_ = (mascara_calzada if mascara_calzada is not None else util).astype(np.uint8)
        mcal_p = np.pad(mcal_, 1, mode="constant", constant_values=1)   #solo el borde real de la calzada, no el recorte de la unidad
        ceb_ = (cv2.distanceTransform(mcal_p, cv2.DIST_L2, 3)[1:-1, 1:-1] * gsd) < cfg.cable_borde_m
        eb_ = np.bincount(ll_[sel_], weights=ceb_[sel_].astype(np.float64), minlength=nl_)
        quitar_ = (tot_ > 0) & (cer_ >= cfg.cable_cerca_frac * np.maximum(tot_, 1)) & ((eb_ < 0.5 * np.maximum(tot_, 1)) | perp_)
        quitar_[0] = False
        if quitar_.any():
            crack_raw &= ~quitar_[ll_]
            sk_raw = skeletonize(crack_raw)
    if cfg.cable_col_activa and sk_raw.any():
        
        fr_ = max(1.0, 0.03 / gsd)
        w3, h3 = int(round(w / fr_)), int(round(h / fr_))
        g3 = gsd * w / w3
        sk3 = (cv2.resize(cv2.dilate(sk_raw.astype(np.uint8), np.ones((3, 3), np.uint8)), (w3, h3), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        sk3 = skeletonize(sk3 > 0).astype(np.uint8)
        lmin3 = int(cfg.cable_col_largo_m / g3)
        segs3 = cv2.HoughLinesP(sk3 * 255, 1, np.pi / 360, threshold=max(10, int(lmin3 * cfg.cable_col_apoyo * 0.6)),
                                minLineLength=lmin3, maxLineGap=int(cfg.cable_col_hueco_m / g3))
        if segs3 is not None:
            ban3 = cv2.dilate(sk3, np.ones((5, 5), np.uint8)) > 0
            _, ll3 = cv2.connectedComponents(sk3, connectivity=8)
            lin3 = np.zeros((h3, w3), np.uint8)
            for x1_, y1_, x2_, y2_ in np.asarray(segs3).reshape(-1, 4):
                L_ = int(math.hypot(x2_ - x1_, y2_ - y1_))
                if L_ < lmin3:
                    continue
                xs_ = np.clip(np.linspace(x1_, x2_, L_).astype(int), 0, w3 - 1)
                ys_ = np.clip(np.linspace(y1_, y2_, L_).astype(int), 0, h3 - 1)
                if ban3[ys_, xs_].mean() < cfg.cable_col_apoyo:
                    continue
                if len(np.unique(ll3[ys_, xs_][ll3[ys_, xs_] > 0])) < cfg.cable_col_frag_min:
                    continue
                
                tmp3 = np.zeros((h3, w3), np.uint8)
                cv2.line(tmp3, (int(x1_), int(y1_)), (int(x2_), int(y2_)), 1, 3)
                n_in = int((sk3 & tmp3).sum())
                tmp3[:] = 0
                cv2.line(tmp3, (int(x1_), int(y1_)), (int(x2_), int(y2_)), 1, int(1.0 / g3))
                n_out = int((sk3 & tmp3).sum()) - n_in
                if n_out > cfg.cable_col_entorno * max(n_in, 1):
                    continue
                cv2.line(lin3, (int(x1_), int(y1_)), (int(x2_), int(y2_)), 1, 1)
            if lin3.any():
                lin_ = cv2.resize(lin3, (w, h), interpolation=cv2.INTER_NEAREST)
                lin_ = cv2.dilate(lin_, _elipse(cfg.cable_col_ancho_m / gsd)).astype(bool)
                mcal_ = (mascara_calzada if mascara_calzada is not None else util).astype(np.uint8)
                mcal_p = np.pad(mcal_, 1, mode="constant", constant_values=1)
                ceb_ = (cv2.distanceTransform(mcal_p, cv2.DIST_L2, 3)[1:-1, 1:-1] * gsd) < cfg.cable_borde_m
                nl_, ll_, st__, _ = cv2.connectedComponentsWithStats(crack_raw.astype(np.uint8), connectivity=8)
                perp_ = st__[:, cv2.CC_STAT_HEIGHT] > 1.5 * st__[:, cv2.CC_STAT_WIDTH]
                sel_ = ll_ > 0
                tot_ = np.bincount(ll_[sel_], minlength=nl_).astype(np.float64)
                on_ = np.bincount(ll_[sel_], weights=lin_[sel_].astype(np.float64), minlength=nl_)
                eb_ = np.bincount(ll_[sel_], weights=ceb_[sel_].astype(np.float64), minlength=nl_)
                quitar_ = (tot_ > 0) & (on_ >= 0.5 * np.maximum(tot_, 1)) & ((eb_ < 0.5 * np.maximum(tot_, 1)) | perp_)
                quitar_[0] = False
                if quitar_.any():
                    crack_raw &= ~quitar_[ll_]
                    sk_raw = skeletonize(crack_raw)
    pico_global = float(np.percentile(-norm_g[sk_raw], 75)) if sk_raw.any() else 1.0

    #piel de cocodrilo
    zonas_piel = np.zeros((h, w), bool)
    zonas: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
    
    for pf in (piel_forma or []):
        zona = pf & borde & ~bache_dil & ~zonas_piel   
        area_m2 = float(zona.sum()) * gsd * gsd
        if area_m2 < cfg.piel_area_min_m2 * 0.2:
            continue
        dens = float((sk_raw & zona).sum()) * gsd * FACTOR_LONGITUD / max(area_m2, 1e-6)
        cob = float(np.clip(dens * 0.012 / 0.25, 0.0, 1.0))
        b_, m_ = cfg.piel_cobertura_sev
        sev = "B" if cob < b_ * 4 else ("M" if cob < m_ * 4 else "A")
        zonas_piel |= zona
        zonas.append((zona, color_falla(1, sev)))
        x, y, bw, bh = cv2.boundingRect(zona.astype(np.uint8))
        instancias.append({"codigo": 1, "severidad": sev, "valor": area_m2, "unidad_medida": "m2",
                           "x": int(x) + offset[0], "y": int(y) + offset[1], "w": int(bw), "h": int(bh),
                           "detalle": f"plantilla; dens={dens:.1f} m/m2"})
    crack_dil = cv2.dilate(crack_raw.astype(np.uint8), np.ones((3, 3), np.uint8))
    libre = (util & (crack_dil == 0) & ~bache_dil).astype(np.uint8)
    nl, lab_l, st_l, cen_l = cv2.connectedComponentsWithStats(libre, connectivity=4)
    amin, amax = cfg.celda_area_min_m2 / gsd ** 2, cfg.celda_area_max_m2 / gsd ** 2
    exterior_d = cv2.dilate((~borde).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    abiertos = set(np.unique(lab_l[exterior_d & (libre > 0)]).tolist())
    ids_celda = [i for i in range(1, nl) if amin <= st_l[i, cv2.CC_STAT_AREA] <= amax and i not in abiertos]
    if len(ids_celda) >= cfg.piel_min_celdas:
        celdas = np.isin(lab_l, ids_celda).astype(np.uint8)
        union = ((celdas > 0) | crack_raw).astype(np.uint8)
        cierre = cv2.morphologyEx(union, cv2.MORPH_CLOSE, _elipse(cfg.piel_cierre_m / gsd))
        nc, lab_c, st_c, _ = cv2.connectedComponentsWithStats(cierre, connectivity=8)
        conteo = np.zeros(nc, int)
        for i in ids_celda:
            cx, cy = cen_l[i]
            conteo[lab_c[int(cy), int(cx)]] += 1
        for k in range(1, nc):
            area_m2 = st_c[k, cv2.CC_STAT_AREA] * gsd * gsd
            if conteo[k] >= cfg.piel_min_celdas and area_m2 >= cfg.piel_area_min_m2:
                zona = lab_c == k
                ys_k, xs_k = np.nonzero(zona)
                (_, _), (rw_, rh_), _ = cv2.minAreaRect(np.column_stack([xs_k, ys_k]).astype(np.float32))
                lado_c, lado_l = min(rw_, rh_) * gsd, max(rw_, rh_) * gsd
                if lado_l > 1.0 and lado_l / max(lado_c, 0.05) > cfg.piel_alargamiento_max:
                    continue
                sk_z = sk_raw & zona
                pico = max(1.0, float(np.percentile(-norm_g[sk_z], 75))) if sk_z.any() else max(1.0, pico_global)
                cob = float(np.clip((-norm_g[zona]).sum() / pico / float(zona.sum()), 0.0, 1.0))
                b, m = cfg.piel_cobertura_sev
                sev = "B" if cob < b else ("M" if cob < m else "A")
                if zonas_piel.any():
                    zona = zona & ~zonas_piel
                    area_m2 = float(zona.sum()) * gsd * gsd
                    if area_m2 < cfg.piel_area_min_m2:
                        continue
                zonas_piel |= zona
                x, y, bw, bh = cv2.boundingRect(zona.astype(np.uint8))
                zonas.append((zona, color_falla(1, sev)))
                instancias.append({"codigo": 1, "severidad": sev, "valor": float(area_m2), "unidad_medida": "m2",
                                   "x": int(x) + offset[0], "y": int(y) + offset[1], "w": int(bw), "h": int(bh),
                                   "detalle": f"celdas={int(conteo[k])}; cobertura_grieta={cob:.2f}"})
    piel_dil = cv2.dilate(zonas_piel.astype(np.uint8), _elipse(0.03 / gsd)).astype(bool)

    #agrietamiento en bloque
    zonas_bloque = np.zeros((h, w), bool)
    r_cierre = max(1.0, cfg.bloque_cierre_m / gsd)
    crack_blk = cv2.dilate(crack_raw.astype(np.uint8), _elipse(r_cierre))
    libre_b = (util & (crack_blk == 0) & ~bache_dil & ~piel_dil).astype(np.uint8)
    nb_, lab_b, st_b, cen_b = cv2.connectedComponentsWithStats(libre_b, connectivity=4)
    amin_b, amax_b = cfg.celda_area_max_m2 / gsd ** 2, cfg.bloque_area_max_m2 / gsd ** 2
    abiertos_b = set(np.unique(lab_b[exterior_d & (libre_b > 0)]).tolist())
    ids_b = [i for i in range(1, nb_) if amin_b < st_b[i, cv2.CC_STAT_AREA] <= amax_b and i not in abiertos_b]
    for af in (ahuell_forma or []):
        zona = af & util
        area_m2 = float(zona.sum()) * gsd * gsd
        if area_m2 < 0.5:
            continue
        sev = cfg.ahuell_sev
        zonas.append((zona, color_falla(15, sev)))
        ys_, xs_ = np.nonzero(zona)
        instancias.append({"codigo": 15, "severidad": sev, "valor": area_m2, "unidad_medida": "m2",
                           "x": int(xs_.min()) + offset[0], "y": int(ys_.min()) + offset[1],
                           "w": int(xs_.max() - xs_.min() + 1), "h": int(ys_.max() - ys_.min() + 1),
                           "detalle": "plantilla; severidad fija (sin perfil de profundidad)"})
    for bf in (bloque_forma or []):
        zona = bf & util & ~bache_dil & ~piel_dil
        area_m2 = float(zona.sum()) * gsd * gsd
        if area_m2 < 0.3:
            continue
        ancho_mm = _ancho_equivalente_mm(norm_g, crack_raw & zona, sk_raw & zona, gsd)
        sev = _sev_ancho(ancho_mm, cfg)
        zonas_bloque |= zona
        zonas.append((zona, color_falla(3, sev)))
        ys_, xs_ = np.nonzero(zona)
        instancias.append({"codigo": 3, "severidad": sev, "valor": area_m2, "unidad_medida": "m2",
                           "x": int(xs_.min()) + offset[0], "y": int(ys_.min()) + offset[1],
                           "w": int(xs_.max() - xs_.min() + 1), "h": int(ys_.max() - ys_.min() + 1),
                           "detalle": f"plantilla; ancho_grieta={ancho_mm:.1f}mm"})
    if len(ids_b) >= cfg.bloque_min_celdas:
        cel = np.isin(lab_b, ids_b)
        grp = cv2.dilate(cel.astype(np.uint8), _elipse(0.15 / gsd))
        ng, lab_g = cv2.connectedComponents(grp, connectivity=8)
        cuenta = np.zeros(ng, int)
        for i in ids_b:
            cx, cy = cen_b[i]
            cuenta[lab_g[int(cy), int(cx)]] += 1
        for g in range(1, ng):
            if cuenta[g] < cfg.bloque_min_celdas:
                continue
            cel_g = (cel & (lab_g == g)).astype(np.uint8)
            zona = cv2.dilate(cel_g, _elipse(r_cierre + 3)).astype(bool) & util
            area_m2 = float(zona.sum()) * gsd * gsd
            if area_m2 < cfg.bloque_area_min_m2:
                continue
            sk_z = sk_raw & zona
            ancho_mm = _ancho_equivalente_mm(norm_g, crack_raw & zona, sk_z, gsd)
            sev = _sev_ancho(ancho_mm, cfg)
            zonas_bloque |= zona
            zonas.append((zona, color_falla(3, sev)))
            ys_, xs_ = np.nonzero(zona)
            instancias.append({"codigo": 3, "severidad": sev, "valor": area_m2, "unidad_medida": "m2",
                               "x": int(xs_.min()) + offset[0], "y": int(ys_.min()) + offset[1],
                               "w": int(xs_.max() - xs_.min() + 1), "h": int(ys_.max() - ys_.min() + 1),
                               "detalle": f"bloques={int(cuenta[g])}; ancho_grieta={ancho_mm:.1f}mm"})
    bloque_dil = cv2.dilate(zonas_bloque.astype(np.uint8), _elipse(0.03 / gsd)).astype(bool)

    #parcheo
    parche_mask = parche_usuario.copy()
    parches_col = [(np.asarray(m, bool), cfg.parche_usuario_sev) for m in (parches_usuario or [])]
    gs = cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.04 / gsd))
    base = util & ~bache_dil & ~piel_dil & ~bloque_dil
    diff = np.zeros((h, w), np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gm = np.hypot(cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3) / 8.0)
    grad_base = float(np.median(gm[util]))
    if base.sum() > 100:
        plano = _plano(gs, base)
        diff = (gs - plano).astype(np.float32)
        cp = ((np.abs(diff) > cfg.parche_dif_min) & base).astype(np.uint8)
        cp = cv2.morphologyEx(cp, cv2.MORPH_OPEN, _elipse(0.05 / gsd))
        cp = cv2.morphologyEx(cp, cv2.MORPH_CLOSE, _elipse(0.08 / gsd))
        npp, lab_p, st_p, _ = cv2.connectedComponentsWithStats(cp, connectivity=8)
        for i in range(1, npp):
            a_m2 = st_p[i, cv2.CC_STAT_AREA] * gsd * gsd
            if a_m2 < cfg.parche_area_min_m2 or a_m2 > 40.0:
                continue
            x, y, bw, bh = st_p[i, :4]
            comp = (lab_p[y:y + bh, x:x + bw] == i).astype(np.uint8)
            cnt, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnt:
                continue
            c = max(cnt, key=cv2.contourArea)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            rect = cv2.contourArea(c) / (rw * rh + 1e-9)
            if rect < cfg.parche_rectangularidad_min:
                continue
            mk = lab_p == i
            banda_b = cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & \
                ~cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            contraste = max(1.0, abs(float(diff[mk].mean())))
            nitidez = (float(np.percentile(gm[banda_b], 85)) - grad_base) / contraste
            if nitidez < cfg.parche_nitidez_min:
                continue
            mk_u8 = mk.astype(np.uint8)
            anillo = (cv2.dilate(mk_u8, _elipse(0.35 / gsd)) > 0) & ~(cv2.dilate(mk_u8, _elipse(0.06 / gsd)) > 0) & util
            dentro = cv2.erode(mk_u8, _elipse(0.04 / gsd)) > 0
            borde_p = (cv2.dilate(mk_u8, _elipse(0.03 / gsd)) > 0) & ~(cv2.erode(mk_u8, _elipse(0.03 / gsd)) > 0)
            if anillo.sum() < 100 or dentro.sum() < 30 or borde_p.sum() < 30:
                continue
            borde_rel = float(np.percentile(gm[borde_p], 70)) / (float(np.percentile(gm[anillo], 70)) + 1e-3)
            tex_rel = float(gm[dentro].mean()) / (float(gm[anillo].mean()) + 1e-3)
            if borde_rel < cfg.parche_borde_min or tex_rel < cfg.parche_tex_min:
                continue
            parche_mask |= mk
            parches_col.append((mk.copy(), None))
            mk_in = cv2.erode(mk.astype(np.uint8), _elipse(0.05 / gsd)).astype(bool)
            a_in = float(mk_in.sum()) * gsd * gsd
            long_cr = float(sk_raw[mk_in].sum()) * gsd * FACTOR_LONGITUD
            dens = long_cr / a_in if a_in > 0.02 else 0.0
            b, m = cfg.parche_grietas_sev
            sev = "B" if dens < b else ("M" if dens < m else "A")
            parches_col[-1] = (parches_col[-1][0], sev)
            instancias.append({"codigo": 11, "severidad": sev, "valor": float(a_m2), "unidad_medida": "m2",
                               "x": int(x) + offset[0], "y": int(y) + offset[1], "w": int(bw), "h": int(bh),
                               "detalle": f"dif_tono={float(diff[mk].mean()):+.0f}; grietas={dens:.2f}m/m2; nitidez={nitidez:.2f}; borde={borde_rel:.2f}; tex={tex_rel:.2f}; rect={rect:.2f}"})
    parche_dil = cv2.dilate(parche_mask.astype(np.uint8), _elipse(0.02 / gsd)).astype(bool)

    #textura exudacion (2) y desprendimiento de agregados (19)
    sig_hp = max(1.0, 0.012 / gsd)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), sig_hp)
    parche_ancho = cv2.dilate(parche_mask.astype(np.uint8), _elipse(0.08 / gsd)).astype(bool)
    n_raw, lab_raw = cv2.connectedComponents(crack_raw.astype(np.uint8), connectivity=8)
    long_raw = np.bincount(lab_raw[sk_raw], minlength=n_raw) * gsd * FACTOR_LONGITUD
    corta = long_raw < 1.0
    corta[0] = False
    crack_larga = crack_raw & ~corta[lab_raw]
    crack_ancho = cv2.dilate(crack_larga.astype(np.uint8), _elipse(0.06 / gsd)).astype(bool)
    vt = util & ~bache_dil & ~piel_dil & ~bloque_dil & ~parche_ancho & ~crack_ancho
    kt = max(5, int(cfg.textura_ventana_m / gsd) | 1)
    num = cv2.boxFilter((hp * hp) * vt, -1, (kt, kt), borderType=cv2.BORDER_REFLECT)
    den = cv2.boxFilter(vt.astype(np.float32), -1, (kt, kt), borderType=cv2.BORDER_REFLECT)
    tex_ok = (den > 0.6) & vt
    tex = np.sqrt(num / np.maximum(den, 1e-3))
    tex_med = float(np.median(tex[tex_ok])) if tex_ok.sum() > 200 else 0.0
    tex_ref = cfg.textura_ref if cfg.textura_ref else tex_med
    zonas_tex = np.zeros((h, w), bool)
    if tex_ref > 0.5 and tex_ok.sum() > 200:
        ratio = tex / tex_ref
        for df in (desprend_forma or []):
            zona = df & borde & ~bache_dil & ~parche_dil & ~zonas_piel & ~zonas_tex
            area_m2 = float(zona.sum()) * gsd * gsd
            if area_m2 < 0.2:
                continue
            zt = zona & tex_ok
            r_med = float(ratio[zt].mean()) if zt.sum() > 50 else 1.0
            b, m = cfg.desprend_ratio_sev
            sev = "B" if r_med < b else ("M" if r_med < m else "A")
            zonas_tex |= zona
            zonas.append((zona, color_falla(19, sev)))
            ys_, xs_ = np.nonzero(zona)
            instancias.append({"codigo": 19, "severidad": sev, "valor": area_m2, "unidad_medida": "m2",
                               "x": int(xs_.min()) + offset[0], "y": int(ys_.min()) + offset[1],
                               "w": int(xs_.max() - xs_.min() + 1), "h": int(ys_.max() - ys_.min() + 1),
                               "detalle": f"plantilla; textura={r_med:.2f}x_ref"})
        cd = ((ratio >= cfg.desprend_ratio_min) & tex_ok).astype(np.uint8)
        cd = cv2.morphologyEx(cd, cv2.MORPH_OPEN, _elipse(0.10 / gsd))
        cd = cv2.morphologyEx(cd, cv2.MORPH_CLOSE, _elipse(0.15 / gsd))
        nd, lab_d, st_d, _ = cv2.connectedComponentsWithStats(cd, connectivity=8)
        for i in range(1, nd):
            area_m2 = st_d[i, cv2.CC_STAT_AREA] * gsd * gsd
            if area_m2 < cfg.desprend_area_min_m2:
                continue
            zona = (lab_d == i) & util & ~bache_dil & ~parche_dil
            zona = cv2.erode(zona.astype(np.uint8), _elipse(0.07 / gsd)).astype(bool)
            area_m2 = float(zona.sum()) * gsd * gsd
            if area_m2 < cfg.desprend_area_min_m2:
                continue
            r_med = float(ratio[zona & tex_ok].mean()) if (zona & tex_ok).any() else cfg.desprend_ratio_min
            b, m = cfg.desprend_ratio_sev
            sev = "B" if r_med < b else ("M" if r_med < m else "A")
            zonas_tex |= zona
            zonas.append((zona, color_falla(19, sev)))
            ys_, xs_ = np.nonzero(zona)
            instancias.append({"codigo": 19, "severidad": sev, "valor": area_m2, "unidad_medida": "m2",
                               "x": int(xs_.min()) + offset[0], "y": int(ys_.min()) + offset[1],
                               "w": int(xs_.max() - xs_.min() + 1), "h": int(ys_.max() - ys_.min() + 1),
                               "detalle": f"textura={r_med:.2f}x_ref"})
        ce = ((diff < -cfg.exud_dif_min) & (ratio <= cfg.exud_textura_max) & tex_ok & ~zonas_tex).astype(np.uint8)
        ce = cv2.morphologyEx(ce, cv2.MORPH_OPEN, _elipse(0.10 / gsd))
        ce = cv2.morphologyEx(ce, cv2.MORPH_CLOSE, _elipse(0.15 / gsd))
        ne, lab_e, st_e, _ = cv2.connectedComponentsWithStats(ce, connectivity=8)
        for i in range(1, ne):
            area_m2 = st_e[i, cv2.CC_STAT_AREA] * gsd * gsd
            if area_m2 < cfg.exud_area_min_m2:
                continue
            zona = (lab_e == i) & util & ~bache_dil & ~parche_dil
            area_m2 = float(zona.sum()) * gsd * gsd
            dif_med = -float(diff[zona].mean())
            b, m = cfg.exud_sev_dif
            sev = "B" if dif_med < b else ("M" if dif_med < m else "A")
            zonas.append((zona, color_falla(2, sev)))
            ys_, xs_ = np.nonzero(zona)
            instancias.append({"codigo": 2, "severidad": sev, "valor": area_m2, "unidad_medida": "m2",
                               "x": int(xs_.min()) + offset[0], "y": int(ys_.min()) + offset[1],
                               "w": int(xs_.max() - xs_.min() + 1), "h": int(ys_.max() - ys_.min() + 1),
                               "detalle": f"oscurecimiento={dif_med:.0f}ng; textura={float(ratio[zona & tex_ok].mean()):.2f}x_ref"})

    
    if grieta_forma:
        zg = np.zeros((h, w), bool)
        for gf_ in grieta_forma:
            zg |= gf_
        zg = cv2.dilate(zg.astype(np.uint8), _elipse(cfg.grieta_zona_margen_m / gsd)).astype(bool)
        crack_raw = crack_raw | (cr & oscuro & zg & ~bache_dil)
    #grietas borde (7), parabolicas (17), longitudinales/transversales (10)
    crack_lin = crack_raw & ~piel_dil & ~bloque_dil & ~parche_dil
    if zonas_tex.any():
        zt_dil = cv2.dilate(zonas_tex.astype(np.uint8), _elipse(0.05 / gsd)).astype(bool)
        crack_lin &= ~(zt_dil & corta[lab_raw])
    sk = skeletonize(crack_lin)
    ncr, lab_cr, st_cr, _ = cv2.connectedComponentsWithStats(crack_lin.astype(np.uint8), connectivity=8)
    mc_ = (mascara if mascara_calzada is None else mascara_calzada).astype(np.uint8).copy()
    
    cap_ = int(3.0 / gsd)
    fil_ = mc_.any(axis=1)
    pri_ = mc_.argmax(axis=1)
    ult_ = mc_.shape[1] - 1 - mc_[:, ::-1].argmax(axis=1)
    col_ = np.arange(mc_.shape[1])[None, :]
    mc_[(fil_ & (pri_ <= cap_))[:, None] & (col_ < pri_[:, None])] = 1
    mc_[(fil_ & (mc_.shape[1] - 1 - ult_ <= cap_))[:, None] & (col_ > ult_[:, None])] = 1
    nh_, lh_ = cv2.connectedComponents((mc_ == 0).astype(np.uint8), connectivity=4)
    if nh_ > 1:
        tocan_ = set(np.unique(np.concatenate([lh_[0, :], lh_[-1, :], lh_[:, 0], lh_[:, -1]])).tolist())
        for i_ in range(1, nh_):
            if i_ not in tocan_:
                mc_[lh_ == i_] = 1
    cols_ = np.nonzero(mc_.any(axis=0))[0]
    if len(cols_) > 0:
        mg_ = max(2, int(1.0 / gsd))
        xa_, xb_ = int(cols_[0]), int(cols_[-1])
        if xb_ - xa_ > 3 * mg_:
            mc_[:, :xa_ + mg_] = mc_[:, [xa_ + mg_]]
            mc_[:, xb_ - mg_ + 1:] = mc_[:, [xb_ - mg_]]
    mp = np.pad(mc_, 1, mode="edge")
    mp[0, :] = 0
    mp[-1, :] = 0
    dist_borde = cv2.distanceTransform(mp, cv2.DIST_L2, 3)[1:-1, 1:-1] * gsd
    capas_grietas: List[Tuple[Tuple[int, int, int, int, np.ndarray], Tuple[int, int, int]]] = []
    for i in range(1, ncr):
        x, y, bw, bh, _ = st_cr[i]
        sub = lab_cr[y:y + bh, x:x + bw] == i
        sk_sub = sk[y:y + bh, x:x + bw] & sub
        n_sk = int(_podar_esqueleto(sk_sub, max(2, int(round(cfg.grieta_espina_m / gsd)))).sum())
        long_m = n_sk * gsd * FACTOR_LONGITUD
        if long_m < cfg.grieta_long_min_m:
            continue
        ancho_mm = _ancho_equivalente_mm(norm_g[y:y + bh, x:x + bw], sub, sk_sub, gsd)
        ys, xs = np.nonzero(sk_sub)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        hull_ = cv2.convexHull(pts).reshape(-1, 2)
        sub_h = hull_[::max(1, len(hull_) // 60)]
        ext_m = float(np.max(np.hypot(sub_h[:, None, 0] - sub_h[None, :, 0], sub_h[:, None, 1] - sub_h[None, :, 1]))) * gsd
        contraste_medio = float((-norm_g[y:y + bh, x:x + bw])[sub].mean())
        if ext_m < cfg.grieta_ext_min_m:
            continue
        if float(np.mean(dist_borde[ys + y, xs + x])) < cfg.borde_artefacto_m:
            continue   
        if contraste_medio > cfg.grieta_contraste_medio_max:
            continue
        vx, vy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
        ang = abs(math.degrees(math.atan2(float(vy), float(vx))))
        tipo = "longitudinal" if (ang <= 45 or ang >= 135) else "transversal"
        sev = _sev_ancho(ancho_mm, cfg)
        base_ins = {"severidad": sev, "x": int(x) + offset[0], "y": int(y) + offset[1], "w": int(bw), "h": int(bh)}

        c0 = pts.mean(axis=0)
        _, _, vt_ = np.linalg.svd(pts - c0, full_matrices=False)
        ue = vt_[0]
        proy = (pts - c0) @ ue
        e1, e2 = pts[np.argmin(proy)], pts[np.argmax(proy)]
        cuerda_px = float(np.linalg.norm(e2 - e1))
        if cuerda_px * gsd >= cfg.parab_cuerda_min_m:
            d_ = (e2 - e1) / cuerda_px
            dperp = (pts[:, 0] - e1[0]) * (-d_[1]) + (pts[:, 1] - e1[1]) * d_[0]
            k_max = int(np.argmax(np.abs(dperp)))
            flecha_px = float(abs(dperp[k_max]))
            mismo_lado = float(np.mean(np.sign(dperp) == np.sign(dperp[k_max])))
            ajuste_ok = False
            if flecha_px / cuerda_px >= cfg.parab_flecha_min and mismo_lado >= 0.9:
                try:
                    cf = np.polyfit(proy, dperp, 2)
                    rms = float(np.sqrt(np.mean((np.polyval(cf, proy) - dperp) ** 2)))
                    ajuste_ok = (rms / cuerda_px <= cfg.parab_rms_max) and (long_m / (cuerda_px * gsd) <= cfg.parab_tortuosidad_max)
                except Exception:
                    ajuste_ok = False
            if ajuste_ok:
                area_m2 = max(0.05, (2.0 / 3.0) * (cuerda_px * gsd) * (flecha_px * gsd))
                bp_, mp_ = cfg.parab_ancho_sev_mm
                sev = "B" if ancho_mm < bp_ else ("M" if ancho_mm < mp_ else "A")
                instancias.append({**base_ins, "severidad": sev, "codigo": 17, "valor": float(area_m2), "unidad_medida": "m2",
                                   "detalle": f"cuerda={cuerda_px*gsd:.2f}m; flecha/cuerda={flecha_px/cuerda_px:.2f}; ancho={ancho_mm:.1f}mm"})
                capas_grietas.append(((int(x), int(y), int(bw), int(bh), sub), color_falla(17, sev)))
                continue

        frac_borde = float(np.mean(dist_borde[ys + y, xs + x] < cfg.borde_grieta_m))
        tortuosidad = long_m / max(ext_m, 1e-6)
        notoria = ancho_mm >= cfg.borde_ancho_min_mm and contraste_medio >= cfg.borde_contraste_min and tortuosidad >= cfg.borde_tortuosidad_min
        if tipo == "longitudinal" and (ang <= cfg.borde_ang_max or ang >= 180 - cfg.borde_ang_max) and frac_borde >= cfg.borde_grieta_frac and notoria:
            instancias.append({**base_ins, "codigo": 7, "valor": float(long_m), "unidad_medida": "m",
                               "detalle": f"longitudinal; en_franja_borde={frac_borde:.0%}; ancho={ancho_mm:.1f}mm; contraste={contraste_medio:.1f}; tort={tortuosidad:.2f}"})
            capas_grietas.append(((int(x), int(y), int(bw), int(bh), sub), color_falla(7, sev)))
            continue

        instancias.append({**base_ins, "codigo": 10, "valor": float(long_m), "unidad_medida": "m",
                           "detalle": f"{tipo}; ancho_medio={ancho_mm:.1f}mm; contraste={contraste_medio:.1f}; borde={frac_borde:.0%}; tort={long_m / max(ext_m, 1e-6):.2f}"})
        capas_grietas.append(((int(x), int(y), int(bw), int(bh), sub), COLOR_SEV[sev]))

    
    if cfg.extra_activa and (dsm is not None or zonas_extra):
        ins_x, capas_x = detectar_complementarias(dsm, zonas_extra, mascara, gsd, cfg, offset)
        instancias += ins_x
        for zona_x, cod_x, sev_x in capas_x:
            zonas.append((zona_x, color_falla(cod_x, sev_x)))

    #agregacion por (codigo, severidad)
    acum: Dict[Tuple[int, str], float] = {}
    for ins in instancias:
        k = (ins["codigo"], ins["severidad"])
        acum[k] = acum.get(k, 0.0) + ins["valor"]
    fallas = [{"codigo": c, "nombre": NOMBRES_FALLA[c], "severidad": s, "cantidad": float(v),
               "unidad_medida": UNIDAD_MEDIDA[c]} for (c, s), v in sorted(acum.items())]

    overlay = None
    if generar_overlay:
        overlay = _dibujar_overlay(img_bgr, zonas, capas_grietas, baches, baches_sev, parches_col, gsd)
    presentes = sorted({(i["codigo"], i["severidad"]) for i in instancias if i["codigo"] in NOMBRE_LEYENDA})
    leyenda = [(f"{NOMBRE_LEYENDA[c]} {s}", color_falla(c, s)) for c, s in presentes]
    return {"fallas": fallas, "instancias": instancias, "overlay": overlay, "leyenda": leyenda,
            "util_m2": float(util.sum()) * gsd * gsd, "textura_mediana": tex_med, "textura_ref": float(tex_ref)}


#visualizacion
COLOR_SEV = {"B": (0, 215, 255), "M": (0, 140, 255), "A": (0, 0, 255)}
COLOR_BASE = {1: (200, 0, 200), 2: (200, 100, 0), 3: (0, 150, 150), 19: (230, 170, 60), 11: (0, 170, 0),
              13: (255, 80, 0), 7: (255, 255, 0), 17: (255, 0, 140), 15: (90, 60, 200),
              4: (150, 150, 0), 5: (0, 100, 200), 6: (160, 60, 60), 8: (60, 200, 200), 9: (120, 120, 255),
              12: (200, 200, 120), 14: (100, 100, 100), 16: (255, 150, 150), 18: (0, 200, 100)}
NOMBRE_LEYENDA = {1: "Piel de cocodrilo", 2: "Exudación", 3: "Bloques", 7: "Grieta de borde", 10: "Grieta long./transv.",
                  11: "Parche", 13: "Bache", 15: "Ahuellamiento", 17: "Grieta parabólica", 19: "Desprendimiento",
                  4: "Abultamiento/hundimiento", 5: "Corrugación", 6: "Depresión", 8: "Reflexión de junta",
                  9: "Desnivel carril/berma", 12: "Pulimento", 14: "Cruce de vía férrea", 16: "Desplazamiento", 18: "Hinchamiento"}


def color_falla(cod: int, sev: str):
    if cod == 10:
        return COLOR_SEV[sev]
    b = np.array(COLOR_BASE[cod], float)
    c = b + (255 - b) * 0.55 if sev == "B" else (b * 0.55 if sev == "A" else b)
    return tuple(int(v) for v in c)


def _dibujar_overlay(img, zonas, capas_grietas, baches, baches_sev, parches_col, gsd, ancho_max=2200):
    out = img.copy()
    capa = img.copy()
    for (zona, color) in zonas:
        capa[zona] = color
    out = cv2.addWeighted(capa, 0.40, out, 0.60, 0)
    for (pm, sev) in parches_col:
        if pm.any():
            cnt, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnt, -1, color_falla(11, sev or "B"), max(2, int(0.03 / gsd)))
    for (x, y, bw, bh, sub), color in capas_grietas:
        pad_ = max(2, int(0.035 / gsd))     #trazo grueso para que la longitud se vea en el informe
        sub_p = np.pad(sub.astype(np.uint8), pad_)
        sub_p = cv2.dilate(sub_p, _elipse(pad_)).astype(bool)
        y0_, x0_ = max(0, y - pad_), max(0, x - pad_)
        y1_, x1_ = min(out.shape[0], y + bh + pad_), min(out.shape[1], x + bw + pad_)
        sp_ = sub_p[y0_ - (y - pad_): sub_p.shape[0] - ((y + bh + pad_) - y1_), x0_ - (x - pad_): sub_p.shape[1] - ((x + bw + pad_) - x1_)]
        out[y0_:y1_, x0_:x1_][sp_] = color
    for (i, x, y, bw, bh, a_m2, diam_mm, _), sev in zip(baches, baches_sev):
        cx, cy = x + bw // 2, y + bh // 2
        cv2.circle(out, (cx, cy), int(max(bw, bh) / 2 + 0.05 / gsd), color_falla(13, sev), max(2, int(0.03 / gsd)))
    escala = min(1.0, ancho_max / out.shape[1])
    if escala < 1.0:
        out = cv2.resize(out, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return out


CODIGOS_MASCARA = (8, 12, 14)
RES_TRABAJO_M = 0.03


def _sev(valor: float, cortes: Tuple[float, float]) -> str:
    return "B" if valor < cortes[0] else ("M" if valor < cortes[1] else "A")


def _gauss(a: np.ndarray, m: np.ndarray, sigma_px: float) -> np.ndarray:
   
    num = cv2.GaussianBlur(np.where(m, a, 0.0).astype(np.float32), (0, 0), sigma_px)
    den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma_px)
    return num / np.maximum(den, 1e-6)


def _componentes(mask: np.ndarray, area_min_px: float):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= area_min_px:
            yield i, lab == i, st[i]


def _ins(codigo: int, sev: str, valor: float, comp: np.ndarray, g: float, off: Tuple[int, int], f: float, detalle: str) -> Dict:
    ys, xs = np.nonzero(comp)
    return {"codigo": codigo, "severidad": sev, "valor": float(valor), "unidad_medida": CATALOGO_ASTM[codigo][1],
            "x": int(xs.min() * f) + off[0], "y": int(ys.min() * f) + off[1],
            "w": int((xs.max() - xs.min() + 1) * f), "h": int((ys.max() - ys.min() + 1) * f), "detalle": detalle, "_zona": comp}


def _dimensiones(comp: np.ndarray, g: float) -> Tuple[float, float]:
    
    ys, xs = np.nonzero(comp)
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    return max(rw, rh) * g, min(rw, rh) * g


def detectar_relieve(dsm: np.ndarray, mascara: np.ndarray, gsd: float, cfg: Config, offset: Tuple[int, int] = (0, 0)) -> List[Dict]:
    
    f = max(1.0, RES_TRABAJO_M / gsd)
    w, h = int(round(dsm.shape[1] / f)), int(round(dsm.shape[0] / f))
    g = gsd * dsm.shape[1] / w
    z = cv2.resize(dsm.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) * 1000.0   #milimetros
    m = cv2.resize(mascara.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if not m.any():
        return []
    px = lambda metros: max(1.0, metros / g)
    fondo = _gauss(z, m, px(cfg.extra_fondo_m))
    res = np.where(m, z - fondo, 0.0).astype(np.float32)
    salida: List[Dict] = []
    a_px = lambda m2: m2 / (g * g)

    #6 depresion hundimiento suave y ancho con profundidad minima de 13 mm
    for _, comp, _st in _componentes((_gauss(res, m, px(0.15)) <= -cfg.extra_prof_min_mm) & m, a_px(cfg.extra_area_min_m2)):
        largo, ancho = _dimensiones(comp, g)
        if ancho < cfg.extra_ancho_depresion_m:
            continue
        prof = float(-res[comp].min())
        salida.append(_ins(6, _sev(prof, cfg.extra_sev_prof_mm), comp.sum() * g * g, comp, g, offset, f, f"profundidad={prof:.0f}mm"))

    #4 abultamientos y hundimientos localizados y estrechos, se mide el largo
    angosto = (np.abs(_gauss(res, m, px(0.10))) >= cfg.extra_prof_min_mm) & m
    for _, comp, _st in _componentes(angosto, a_px(cfg.extra_area_min_m2)):
        largo, ancho = _dimensiones(comp, g)
        if ancho >= cfg.extra_ancho_depresion_m:
            continue
        amp = float(np.abs(res[comp]).max())
        salida.append(_ins(4, _sev(amp, cfg.extra_sev_prof_mm), largo, comp, g, offset, f, f"amplitud={amp:.0f}mm"))

    #18 hinchamiento 
    onda = _gauss(z, m, px(1.0)) - _gauss(z, m, px(4.0))
    for _, comp, _st in _componentes((onda >= cfg.extra_hinchamiento_mm) & m, a_px(1.0)):
        amp = float(onda[comp].max())
        salida.append(_ins(18, _sev(amp, cfg.extra_sev_prof_mm), comp.sum() * g * g, comp, g, offset, f, f"altura={amp:.0f}mm"))

    #5 corrugacion ondulacion periodica a lo largo de la via 
    banda = _gauss(res, m, px(0.05)) - _gauss(res, m, px(0.40))
    energia = np.sqrt(np.maximum(_gauss(banda ** 2, m, px(0.60)), 0.0))
    for _, comp, _st in _componentes((energia >= cfg.extra_corrug_mm) & m, a_px(1.0)):
        cols = np.nonzero(comp.any(axis=0))[0]
        perfil = np.array([banda[comp[:, c], c].mean() for c in cols])
        cruces = int(np.sum(np.diff(np.sign(perfil)) != 0))
        if cruces < 4:
            continue
        amp = float(energia[comp].max() * math.sqrt(2.0))
        salida.append(_ins(5, _sev(amp, cfg.extra_sev_corrug_mm), comp.sum() * g * g, comp, g, offset, f, f"amplitud={amp:.0f}mm"))

    #16 desplazamiento dipolo de empuje a lo largo de la via
    pos = (_gauss(res, m, px(0.10)) >= cfg.extra_prof_min_mm * 0.8) & m
    neg = (_gauss(res, m, px(0.10)) <= -cfg.extra_prof_min_mm * 0.8) & m
    cerca = cv2.dilate(neg.astype(np.uint8), np.ones((int(px(0.5)) | 1, int(px(0.5)) | 1), np.uint8)) > 0
    for _, comp, _st in _componentes(pos & cerca, a_px(cfg.extra_area_min_m2)):
        amp = float(res[comp].max())
        salida.append(_ins(16, _sev(amp, cfg.extra_sev_prof_mm), comp.sum() * g * g, comp, g, offset, f, f"cresta={amp:.0f}mm"))

    #9 desnivel carril/berma 
    ancho_px = int(px(0.5))
    dentro = m & ~(cv2.erode(m.astype(np.uint8), np.ones((2 * ancho_px + 1, 2 * ancho_px + 1), np.uint8)) > 0)
    fuera = (cv2.dilate(m.astype(np.uint8), np.ones((2 * ancho_px + 1, 2 * ancho_px + 1), np.uint8)) > 0) & ~m
    if dentro.any() and fuera.any():
        zi = _gauss(z, dentro, px(0.30))
        zo = _gauss(z, fuera, px(0.30))
        escalon = np.abs(zi - zo)
        cand = dentro & (escalon >= cfg.extra_desnivel_mm)
        for _, comp, _st in _componentes(cand, a_px(0.05)):
            largo, _ = _dimensiones(comp, g)
            alto = float(escalon[comp].max())
            salida.append(_ins(9, _sev(alto, cfg.extra_sev_desnivel_mm), largo, comp, g, offset, f, f"escalon={alto:.0f}mm"))
    return salida


def detectar_desde_mascara(codigo: int, zona: np.ndarray, mascara: np.ndarray, gsd: float, cfg: Config, offset: Tuple[int, int] = (0, 0)) -> List[Dict]:
    #8 se mide en metros de esqueleto, 12 y 14 en m2. la severidad no sale de la imagen se toma de cfg.extra_sev_defecto
    f = max(1.0, RES_TRABAJO_M / gsd)
    w, h = int(round(zona.shape[1] / f)), int(round(zona.shape[0] / f))
    g = gsd * zona.shape[1] / w
    z = (cv2.resize(zona.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0) & \
        (cv2.resize(mascara.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    salida: List[Dict] = []
    for _, comp, _st in _componentes(z, 4):
        if codigo == 8:
            valor = float(skeletonize(comp).sum()) * g * 1.1
        else:
            valor = float(comp.sum()) * g * g
        salida.append(_ins(codigo, cfg.extra_sev_defecto, valor, comp, g, offset, f, "desde mascara aportada por el usuario"))
    return salida


def detectar_complementarias(dsm, zonas, mascara: np.ndarray, gsd: float, cfg: Config, offset: Tuple[int, int] = (0, 0)):
    
    ins: List[Dict] = []
    if dsm is not None:
        ok = np.isfinite(dsm)
        ins += detectar_relieve(np.where(ok, dsm, 0.0), mascara & ok, gsd, cfg, offset)
    for cod, zona in (zonas or {}).items():
        if cod in CODIGOS_MASCARA:
            ins += detectar_desde_mascara(cod, zona, mascara, gsd, cfg, offset)
    h, w = mascara.shape
    capas = []
    for i in ins:
        z = i.pop("_zona")
        capas.append((cv2.resize(z.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0, i["codigo"], i["severidad"]))
    return ins, capas
