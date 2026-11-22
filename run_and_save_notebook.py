#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner script to execute all code cells and compile ProjekAkhir.ipynb with full outputs.
"""
import sys
import os
import io
import json
import base64
import time
import traceback
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

import generate_cells_part1 as p1
import generate_cells_part2 as p2
import generate_cells_part3 as p3
import generate_cells_part4 as p4

def main():
    print("=" * 80)
    print("MEMULAI PROSES PENYUSUNAN & EKSEKUSI REVISI ProjekAkhir.ipynb")
    print("=" * 80)
    
    # 1. Kumpulkan seluruh sel
    cells = p1.get_notebook_cells()
    p2.add_fase_2_and_3_cells(cells)
    p3.add_fase_4_and_5_cells(cells)
    p4.add_fase_6_and_7_cells(cells)
    
    print(f"Total sel yang dikumpulkan: {len(cells)} sel")
    
    # 2. Setup namespace eksekusi global
    execution_namespace = {}
    
    # Buat custom display function agar display(df) mencetak tabel rapi
    captured_displays = []
    
    def notebook_display(*args):
        for a in args:
            if hasattr(a, 'to_string'):
                print(a.to_string())
            elif hasattr(a, '__str__'):
                print(str(a))
            else:
                print(repr(a))

    execution_namespace['display'] = notebook_display
    
    exec_count = 1
    
    # Iterasi setiap sel
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            continue
            
        code_source = "".join(cell['source'])
        print(f"\n[{exec_count:02d}/25] Menjalankan Sel Kode #{idx}...")
        header_line = [l for l in code_source.split('\n') if l.startswith('# 2.') or l.startswith('# 3.') or l.startswith('# 4.') or l.startswith('# 5.') or l.startswith('# 6.')]
        if header_line:
            print(f"     Substansi: {header_line[0]}")
            
        cell['execution_count'] = exec_count
        cell['outputs'] = []
        
        # Tangkap stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_out = io.StringIO()
        redirected_err = io.StringIO()
        sys.stdout = redirected_out
        sys.stderr = redirected_err
        
        # Bersihkan plot sebelumnya
        plt.close('all')
        
        start_t = time.time()
        error_occurred = False
        try:
            exec(code_source, execution_namespace)
        except Exception as e:
            error_occurred = True
            tb = traceback.format_exc()
            print(f"ERROR TERJADI: {tb}", file=sys.stderr)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        elapsed = time.time() - start_t
        out_text = redirected_out.getvalue()
        err_text = redirected_err.getvalue()
        
        if error_occurred:
            print(f"     ❌ GAGAL dalam {elapsed:.2f}s!")
            print(err_text)
            raise RuntimeError(f"Gagal mengeksekusi sel #{idx}:\n{err_text}")
        else:
            print(f"     ✓ Sukses dalam {elapsed:.2f}s")
            
        # Simpan stream output jika ada
        combined_text = out_text
        if err_text:
            combined_text += "\n[stderr]:\n" + err_text
            
        if combined_text.strip():
            # Format text as list of lines with newlines
            lines = [line + "\n" for line in combined_text.split("\n")]
            if lines and lines[-1] == "\n":
                lines = lines[:-1]
            cell['outputs'].append({
                "name": "stdout",
                "output_type": "stream",
                "text": lines
            })
            
        # Tangkap figures dari matplotlib
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            
            cell['outputs'].append({
                "data": {
                    "image/png": img_b64,
                    "text/plain": [f"<Figure size {fig.get_size_inches()[0]*120:.0f}x{fig.get_size_inches()[1]*120:.0f} with {len(fig.axes)} Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
        plt.close('all')
        
        exec_count += 1
        
    # 3. Format Notebook JSON (v4.4 standard)
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (ds_project_praktikum)",
                "language": "python",
                "name": "ds_project_praktikum"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": sys.version.split()[0]
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # 4. Tulis ke ProjekAkhir.ipynb
    target_nb_path = "ProjekAkhir.ipynb"
    print(f"\nMenyimpan seluruh sel dan output ke '{target_nb_path}'...")
    with open(target_nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
        
    file_size_kb = os.path.getsize(target_nb_path) / 1024
    print(f"🎉 SUKSES BESAR! File '{target_nb_path}' berhasil disimpan!")
    print(f"Ukuran file: {file_size_kb:.2f} KB")
    print(f"Total sel: {len(cells)} ({exec_count-1} kode + {len(cells)-(exec_count-1)} markdown)")

if __name__ == "__main__":
    main()
