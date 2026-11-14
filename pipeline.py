#flujo completo ortomosaico 
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from . import ortomosaico as om
from .basedatos import BaseDatos
from .config import Config
from .deteccion import CODIGOS_MASCARA, detectar_unidad
from .indicadores import calcular_indicadores
from .pci import MotorPCI, clasificar_pci, pci_seccion
from .registro import LOGGER, Cronometro, configurar_logging
from . import reporte


def analizar(ruta_ortomosaico: str, cfg: Config, salida: str, ruta_curvas: Optional[str] = None,
             nombre: str = "Evaluación PCI con ortomosaico", generar_pdf: bool = True) -> Dict:
    os.makedirs(salida, exist_ok=True)
    configurar_logging(salida)
    crono = Cronometro()
    motor = MotorPCI(cfg, ruta_curvas)
    LOGGER.info("Curvas VD/VDC: %s", motor.fuente_curvas)
    cfg.guardar_json(os.path.join(salida, "parametros.json"))
    advertencias: List[str] = []

    #modulo 1 captura de datos
    with crono.medir("captura_datos"):
        fuente = om.abrir_ortomosaico(ruta_ortomosaico, cfg.gsd_m)
        gsd = fuente.gsd
        if gsd is None:
            raise ValueError("No se conoce el GSD (m/píxel). Use --gsd, un JSON hermano {\"gsd_m\": ...} "
                             "o un GeoTIFF en CRS proyectado (metros).")
        LOGGER.info("Ortomosaico %dx%d px | GSD = %.4f m/px (%s) | %s", fuente.ancho, fuente.alto, gsd,
                    fuente.info.get("gsd_origen"), fuente.info.get("lector"))
        if gsd > cfg.gsd_max_recomendado_m:
            msg = (f"GSD = {gsd*1000:.1f} mm/px (> {cfg.gsd_max_recomendado_m*1000:.0f} mm/px): las grietas finas pueden "
                   "no detectarse y la severidad es aproximada.")
            LOGGER.warning(msg)
            advertencias.append(msg)

    #modulo 2 procesamiento de imagenes
    with crono.medir("procesamiento_imagenes"):
        mascara_low = None
        calzada = None
        excluir = None
        limitar = None
        if not cfg.eje_vector:
            raise ValueError("Se requiere --eje (linea del eje dibujada en QGIS).")
        if not cfg.unidades_vector and not cfg.calzada_vector:
            raise ValueError("Sin --unidades-vector la delimitacion es automatica y requiere --calzada (poligono de la calzada).")
        if cfg.excluir_vector:
            excluir = om.calzada_desde_vector(fuente, cfg.excluir_vector)
            LOGGER.info("Zonas excluidas (vehículos, árboles, etc.): %d polígonos", len(excluir))
        parches = None
        if cfg.parches_vector:
            rutas_p = [cfg.parches_vector] if isinstance(cfg.parches_vector, str) else list(cfg.parches_vector)
            parches = []
            for rp in rutas_p:
                parches += om.calzada_desde_vector(fuente, rp)
            LOGGER.info("Parches dibujados por el usuario (falla 11): %d polígonos, severidad %s", len(parches),
                        cfg.parche_usuario_sev)
        dsm = None
        zonas_extra = None
        if cfg.extra_activa:
            if cfg.extra_dsm:
                dsm = om.FuenteDSM(cfg.extra_dsm, fuente)
                LOGGER.info("Detección complementaria con modelo de superficie: %s", cfg.extra_dsm)
            if cfg.extra_mascaras:
                zonas_extra = {}
                for c_, r_ in cfg.extra_mascaras.items():
                    if int(c_) in CODIGOS_MASCARA:
                        zonas_extra[int(c_)] = om.calzada_desde_vector(fuente, r_)
                        LOGGER.info("Detección complementaria falla %d con la capa: %s", int(c_), r_)
                    else:
                        LOGGER.warning("La falla %s no usa capas aportadas: se omite %s", c_, r_)
        Pfull = om.eje_desde_vector(fuente, cfg.eje_vector, cfg.invertir)
        img_low, _, sx, sy, gsd_low, origen = om.lectura_reducida_eje(fuente, gsd, Pfull)
        if cfg.unidades_vector:
            anillos = om.unidades_desde_vector(fuente, cfg.unidades_vector)
            unidades, ancho, largo, P, s_px, total_m, eje_low = om.definir_unidades_poligonos(
                Pfull, anillos, gsd, cfg, origen, sx, sy)
            calzada = [(u["poligono"], []) for u in unidades]
            if cfg.calzada_vector:
                limitar = om.calzada_desde_vector(fuente, cfg.calzada_vector)
                LOGGER.info("Cada unidad se recorta con la calzada dibujada (%d polígonos)", len(limitar))
            LOGGER.info("Unidades de muestra dibujadas por el usuario: %d (de %d contornos leídos)", len(unidades), len(anillos))
            mascara_low = om.mascara_calzada_low(img_low.shape, calzada, origen, sx, sy)
            LOGGER.info("Eje dibujado por el usuario: %.0f m | ancho de calzada %.2f m (promedio, de la calzada dibujada) | largo de unidad %.2f m | %d unidades",
                        total_m, ancho, largo, len(unidades))
        else:
            calzada = om.calzada_desde_vector(fuente, cfg.calzada_vector)
            unidades, ancho, largo, P, s_px, total_m, eje_low = om.definir_unidades_eje(
                Pfull, gsd_low, sx, sy, origen, gsd, cfg, calzada)
            mascara_low = om.mascara_calzada_low(img_low.shape, calzada, origen, sx, sy)
            LOGGER.info("Delimitación automática: eje %.0f m | ancho de calzada %.2f m (promedio) | unidades de %.1f m, una de cada %d | %d unidades",
                        total_m, ancho, largo, cfg.muestreo_cada, len(unidades))
        seleccion = _parsear_unidades(cfg.unidades_sel)
        if seleccion:
            LOGGER.info("Se evaluarán solo las unidades: %s", sorted(seleccion))
        unidades_ok, detecciones = [], []
        for u in unidades:
            if seleccion and u["idx"] not in seleccion:
                continue
            cal_u = [(u["poligono"], [])] if "poligono" in u else calzada
            crop = om.extraer_unidad(fuente, u, P, s_px, mascara_low, sx, sy, gsd, cal_u, excluir, limitar, parches, dsm, zonas_extra)
            if crop is None or crop["area_m2"] <= 0:
                LOGGER.warning("Unidad %d sin calzada: se omite.", u["idx"])
                continue
            if cfg.volcar_recortes:
                os.makedirs(cfg.volcar_recortes, exist_ok=True)
                reg = np.any(crop["parches"], axis=0) if crop.get("parches") else np.zeros_like(crop["mascara"])
                np.savez_compressed(os.path.join(cfg.volcar_recortes, "u%02d.npz" % u["idx"]), img=crop["img"],
                                    mascara=crop["mascara"], excl=crop["excluida_manual"], reg=reg, gsd=gsd)
            mascara_u = crop["mascara"]
            
            ex_m = crop.get("excluida_manual")
            if ex_m is not None and ex_m.any() and cfg.excluir_libre_margen_m > 0:
                k_ = max(1, int(round(cfg.excluir_libre_margen_m / gsd)))
                libre = cv2.distanceTransform((~ex_m).astype(np.uint8), cv2.DIST_L2, 3) <= k_
                mascara_u = mascara_u & ~libre
            parches_u = list(crop.get("parches") or [])
            parches_u = _unir_solapados(parches_u)
            lim = None
            if cfg.auto_limpieza:
                from .limpieza import auto_limpieza
                lim = auto_limpieza(crop["img"], mascara_u, gsd, cfg)
                for cx, cy, rr, _sc in lim["registros"]:
                    c_ = np.zeros(mascara_u.shape, np.uint8)
                    cv2.circle(c_, (int(round(cx)), int(round(cy))), int(round(rr)), 1, -1)
                    c_ = (c_ > 0) & mascara_u
                    if c_.sum() * gsd * gsd < 0.3:
                        continue
                    if any((c_ & q).sum() > 0.3 * c_.sum() for q in parches_u):
                        continue
                    ob_ = lim["partes"].get("objetos")
                    if _sc < 10.0 and ob_ is not None and (c_ & ob_).sum() > 0.2 * c_.sum():
                        continue
                    parches_u.append(c_)
                for pm_ in lim.get("parches_forma", []):
                    c_ = pm_ & mascara_u
                    if c_.sum() * gsd * gsd < 0.05:
                        continue
                    if any((c_ & q).sum() > 0.3 * c_.sum() for q in parches_u):
                        continue
                    parches_u.append(c_)
                excl_auto = lim["excluir"] & ~np.any(parches_u, axis=0) if parches_u else lim["excluir"]
                area_antes = float(mascara_u.sum()) * gsd * gsd
                mascara_u = mascara_u & ~excl_auto
                LOGGER.info("Unidad %d: limpieza automática excluye %.1f m2 (%s) y detecta %d registros", u["idx"],
                            area_antes - float(mascara_u.sum()) * gsd * gsd,
                            ", ".join("%s %.1f" % (k_, float((v_ & crop["mascara"]).sum()) * gsd * gsd) for k_, v_ in lim["partes"].items()),
                            len(lim["registros"]))
            d = detectar_unidad(crop["img"], mascara_u, gsd, cfg, parches_usuario=parches_u, mascara_calzada=crop["mascara"],
                                lineas_auto=(lim["partes"].get("lineas") if lim is not None else None),
                                piel_forma=([pm_ & mascara_u for pm_ in lim.get("piel_forma", [])] if lim is not None else None),
                                desprend_forma=([pm_ & mascara_u for pm_ in lim.get("desprend_forma", [])] if lim is not None else None),
                                bache_forma=([pm_ & mascara_u for pm_ in lim.get("bache_forma", [])] if lim is not None else None),
                                bloque_forma=([pm_ & mascara_u for pm_ in lim.get("bloque_forma", [])] if lim is not None else None),
                                grieta_forma=([pm_ & mascara_u for pm_ in lim.get("grieta_forma", [])] if lim is not None else None),
                                ahuell_forma=([pm_ & mascara_u for pm_ in lim.get("ahuell_forma", [])] if lim is not None else None),
                                dsm=crop.get("dsm"), zonas_extra=crop.get("zonas_extra"))
            if lim is not None and d.get("overlay") is not None:
                ov = d["overlay"]
                ea = cv2.resize(excl_auto.astype(np.uint8), (ov.shape[1], ov.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                ov[ea] = (0.7 * ov[ea] + 0.3 * np.array([200, 200, 200])).astype(np.uint8)
            if cfg.overlay_solo_analizado and d.get("overlay") is not None:
                ov = d["overlay"]
                an = cv2.resize(mascara_u.astype(np.uint8), (ov.shape[1], ov.shape[0]), interpolation=cv2.INTER_NEAREST)
                fuera = an == 0
                #fuera del area analizada solo un velo blanquecino casi transparente 
                ov[fuera] = (0.75 * ov[fuera] + 0.25 * np.array([235, 235, 235])).astype(np.uint8)
                um = crop.get("mascara_um")
                um = cv2.resize(um.astype(np.uint8), (ov.shape[1], ov.shape[0]), interpolation=cv2.INTER_NEAREST) if um is not None else an
                cnt, _ = cv2.findContours(um, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                cv2.drawContours(ov, cnt, -1, (20, 20, 20), 5)
                cv2.drawContours(ov, cnt, -1, (255, 255, 255), 2)
            u["area_analizada_m2"] = float(mascara_u.sum()) * gsd * gsd
            u["area_m2"] = float(crop["area_total_m2"])
            u["x0"], u["y0"], u["x1"], u["y1"] = crop["bbox"]
            unidades_ok.append(u)
            detecciones.append(d)
            LOGGER.info("Unidad %d (%.0f-%.0f m, area total %.0f m2, analizada %.0f m2): %d instancias | textura mediana = %.2f (ref. %.2f)", u["idx"],
                        u["prog_ini_m"], u["prog_fin_m"], u["area_m2"], u["area_analizada_m2"], len(d["instancias"]),
                        d.get("textura_mediana", 0.0), d.get("textura_ref", 0.0))
            del crop
        unidades = unidades_ok
        if not unidades:
            raise RuntimeError("No se generaron unidades de muestra. Revise la máscara de la calzada.")

    ancho = float(np.mean([u["area_m2"] for u in unidades])) / max(largo, 1e-6)
    db_ruta = os.path.join(salida, "resultados.sqlite")
    #modulo 3 calculo del pci
    with crono.medir("calculo_icp", n_items=len(unidades)):
        db = BaseDatos(db_ruta)
        pid = db.crear_proyecto(nombre, os.path.basename(ruta_ortomosaico), gsd, ancho, largo, cfg.a_dict(), motor.fuente_curvas)
        resultados: List[Dict] = []
        for u, d in zip(unidades, detecciones):
            r = motor.calcular_unidad(d["fallas"], u["area_m2"])
            filas = [{"codigo": f["codigo"], "nombre": f["nombre"], "severidad": f["severidad"],
                      "cantidad": f["cantidad"], "unidad_medida": f["unidad_medida"],
                      "densidad_pct": f["densidad_pct"], "vd": f["vd"]} for f in r["detalle"]]
            db.guardar_unidad(pid, u, filas, d["instancias"], r)
            largo_u = float(u["s1_m"] - u["s0_m"])
            resultados.append({**{k: u[k] for k in ("idx", "prog_ini_m", "prog_fin_m", "area_m2", "area_analizada_m2")},
                               "largo_m": largo_u, "ancho_m": float(u["area_m2"]) / largo_u if largo_u > 0 else 0.0,
                               **{k: r[k] for k in ("tdv", "cdv_max", "q", "pci", "clasificacion", "intervencion")},
                               "fallas": filas, "instancias": d["instancias"]})
            LOGGER.info("Unidad %d: PCI = %.1f (%s)", u["idx"], r["pci"], r["clasificacion"])
        pci_t = pci_seccion([r["pci"] for r in resultados], [r["area_m2"] for r in resultados])

    ruta_pdf = os.path.join(salida, "informe_pci.pdf")
    #modulo 4 generacion del reporte
    with crono.medir("generacion_reporte"):
        _exportar_tablas(resultados, salida)
        reporte.generar_excel(os.path.join(salida, "resultados.xlsx"), resultados, motor)
        reporte.guardar_imagenes(img_low, mascara_low, unidades, detecciones, resultados, salida)
        if generar_pdf:
            reporte.generar_pdf(ruta_pdf, nombre, ruta_ortomosaico, cfg, gsd, ancho, largo, resultados, pci_t, motor,
                                salida, advertencias)

    db.guardar_tiempos(pid, crono.registros)
    indicadores = calcular_indicadores(resultados, db, crono, cfg, ruta_pdf, salida)
    with open(os.path.join(salida, "indicadores.json"), "w", encoding="utf-8") as f:
        json.dump(indicadores, f, indent=2, ensure_ascii=False, default=float)
    db.cerrar()

    LOGGER.info("PCI del tramo (ponderado por área) = %.1f (%s) -> %s", pci_t, clasificar_pci(pci_t), motor.intervencion(pci_t))
    return {"resultados": resultados, "pci_tramo": pci_t, "clasificacion_tramo": clasificar_pci(pci_t),
            "intervencion_tramo": motor.intervencion(pci_t), "gsd": gsd, "ancho_via_m": ancho, "largo_unidad_m": largo,
            "salida": salida, "indicadores": indicadores, "advertencias": advertencias}


def _unir_solapados(masks):
    masks = [m.copy() for m in masks]
    cambio = True
    while cambio and len(masks) > 1:
        cambio = False
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if (masks[i] & masks[j]).any():
                    masks[i] |= masks[j]
                    del masks[j]
                    cambio = True
                    break
            if cambio:
                break
    return masks


def _parsear_unidades(txt: Optional[str]):
    if not txt:
        return set()
    out = set()
    for parte in str(txt).replace(" ", "").split(","):
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(parte))
    return out


def _exportar_tablas(resultados: List[Dict], salida: str):
    res = pd.DataFrame([{
        "unidad": r["idx"], "prog_ini_m": round(r["prog_ini_m"], 1), "prog_fin_m": round(r["prog_fin_m"], 1),
        "largo_m": round(r["largo_m"], 2), "ancho_medio_m": round(r["ancho_m"], 2),
        "area_total_m2": round(r["area_m2"], 1), "area_analizada_m2": round(r["area_analizada_m2"], 1), "tdv": round(r["tdv"], 1), "cdv_max": round(r["cdv_max"], 1),
        "pci": round(r["pci"], 1), "clasificacion": r["clasificacion"], "intervencion": r["intervencion"]}
        for r in resultados])
    fal = pd.DataFrame([{"unidad": r["idx"], **f} for r in resultados for f in r["fallas"]])
    ins = pd.DataFrame([{"unidad": r["idx"], **i} for r in resultados for i in r["instancias"]])
    res.to_csv(os.path.join(salida, "resumen_unidades.csv"), index=False, encoding="utf-8-sig")
    fal.to_csv(os.path.join(salida, "fallas_por_unidad.csv"), index=False, encoding="utf-8-sig")
    ins.to_csv(os.path.join(salida, "instancias_fallas.csv"), index=False, encoding="utf-8-sig")
