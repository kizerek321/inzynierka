import os
import sys
import torch
import importlib

# 1. Zarządzanie ścieżkami - kluczowe przy fuzji
PROJECT_ROOT = os.getcwd()
submodules = ["SAMPolyBuild", "Pix2Poly"]

for sub in submodules:
    path = os.path.join(PROJECT_ROOT, sub)
    if path not in sys.path:
        sys.path.append(path)
        print(f"[INFO] Dodano do ścieżki: {sub}")

def test_sam_part():
    print("\n--- Test części SAMPolyBuild ---")
    try:
        from segment_anything import build_sam
        sam_checkpoint = "SAMPolyBuild/segment_anything/sam_vit_b_01ec64.pth"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam(checkpoint=sam_checkpoint, device=device)
        print(f"[OK] SAM (vit_b) zainicjalizowany na {device}")
        
        # Test wag specyficznych
        trained = "SAMPolyBuild/auto_whumix.pth"
        if os.path.exists(trained):
            torch.load(trained, map_location=device)
            print("[OK] Wagi auto_whumix wczytane")
    except Exception as e:
        print(f"[BŁĄD SAM] {e}")

def test_pix2poly_part():
    print("\n--- Test części Pix2Poly ---")
    try:
        # Sprawdzenie zależności
        import timm
        import transformers
        print(f"[OK] Biblioteki obecne (timm: {timm.__version__}, transformers: {transformers.__version__})")
        
        # Test wczytania wag (podaj swoją ścieżkę do wypakowanego folderu runs)
        # Przykładowa ścieżka:
        p2p_checkpoint = "Pix2Poly/runs/Pix2Poly_whu_building_224_coco/logs/checkpoints/epoch_499.pth"
        
        if os.path.exists(p2p_checkpoint):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(p2p_checkpoint, map_location=device)
            print(f"[OK] Checkpoint Pix2Poly wczytany poprawnie (klucze: {list(ckpt.keys())[:3]}...)")
        else:
            print("[!] UWAGA: Nie znaleziono pliku wag Pix2Poly w 'Pix2Poly/runs/'. Sprawdź ścieżkę.")
            
    except Exception as e:
        print(f"[BŁĄD Pix2Poly] {e}")

if __name__ == "__main__":
    print(f"Środowisko: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    
    test_sam_part()
    test_pix2poly_part()
    
    print("\n--- Koniec testu ---")