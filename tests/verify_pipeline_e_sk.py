"""
Pipeline E-SK 동작 검증 스크립트
실제 이미지를 로드하고 Pipeline E-SK를 적용하여 결과를 확인하고 저장합니다.
Pipeline E-SK는 골격화된 경계(skeleton) 검출 기반 전처리를 수행합니다.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hpe.data.loader import load_image_pairs_from_csv, load_image
from src.hpe.pipelines.pipeline_e_sk import process
from src.hpe.pipelines.pipeline_e import process as pipeline_e_process


def verify_pipeline_e_sk(
    save_outputs: bool = True,
    skeleton_method: str = "all",
    compare_with_pipeline_e: bool = True
):
    """Pipeline E-SK의 동작을 실제 이미지로 검증
    
    Args:
        save_outputs: True인 경우 처리된 이미지를 output 폴더에 저장
        skeleton_method: 골격화 방법 ('zhang_suen', 'morphological', 'medial_axis', 또는 'all')
        compare_with_pipeline_e: True인 경우 Pipeline E 결과와 비교
    """
    
    print("=" * 70)
    print(f"Pipeline E-SK 동작 검증 시작")
    print(f"  skeleton_method={skeleton_method}")
    print(f"  compare_with_pipeline_e={compare_with_pipeline_e}")
    print("=" * 70)
    
    # Output 폴더 생성
    output_dir = None
    if save_outputs:
        output_dir = project_root / "output" / "pipeline_e_sk"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n출력 폴더: {output_dir}")
    
    # 1. CSV 파일에서 이미지 쌍 로드
    csv_path = project_root / "tests" / "fixtures" / "test_pipeline.csv"
    print(f"\n1. CSV 파일 로드: {csv_path}")
    
    try:
        pairs = load_image_pairs_from_csv(str(csv_path))
        print(f"   ✓ {len(pairs)}개의 이미지 쌍을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"   ✗ CSV 로드 실패: {e}")
        return False
    
    # 2. 이미지 쌍 처리
    all_passed = True
    
    print(f"\n{'=' * 70}")
    print(f"2. Pipeline E-SK 적용")
    if compare_with_pipeline_e:
        print(f"   (Pipeline E 결과와 비교)")
    print("-" * 70)
    
    for idx, pair in enumerate(pairs, 1):
        source_path = pair["source_path"]
        target_path = pair["target_path"]
        pair_id = pair["image_pair_id"]
        
        print(f"\n[{idx}/{len(pairs)}] 이미지 쌍 처리: {pair_id}")
        print(f"   Source: {source_path}")
        print(f"   Target: {target_path}")
        
        # Source 이미지 처리
        try:
            source_img = load_image(str(project_root / source_path), grayscale=False)
            print(f"   - Source 로드: {source_img.shape}, dtype={source_img.dtype}")
            
            # Pipeline E 결과 (비교용)
            source_e = None
            if compare_with_pipeline_e:
                source_e = pipeline_e_process(
                    source_img,
                    edge_method="sobel",
                    filter_method="connected_components",
                    min_component_size=70
                )
            
            # Pipeline E-SK 처리
            if skeleton_method == "all":
                # 세 가지 방법 모두 적용
                source_results = process(source_img, skeleton_method="all")
                print(f"   - Source 처리 완료: 3가지 방법")
                
                # 각 방법별 통계 출력
                print(f"     통계 비교:")
                if source_e is not None:
                    edge_pixels_e = np.sum(source_e == 255)
                    edge_ratio_e = 100 * edge_pixels_e / source_e.size
                    print(f"       Pipeline E: 에지={edge_pixels_e:,} ({edge_ratio_e:.2f}%)")
                
                for method_name, skeleton in source_results.items():
                    skeleton_pixels = np.sum(skeleton == 255)
                    skeleton_ratio = 100 * skeleton_pixels / skeleton.size
                    
                    # Connected components 개수
                    from scipy import ndimage
                    _, num_components = ndimage.label(skeleton == 255)
                    
                    reduction = 0
                    if source_e is not None:
                        reduction = 100 * (edge_pixels_e - skeleton_pixels) / max(edge_pixels_e, 1)
                    
                    print(f"       {method_name:15s}: 에지={skeleton_pixels:,} ({skeleton_ratio:.2f}%), "
                          f"컴포넌트={num_components:,}, 감소율={reduction:.1f}%")
                
                # 검증
                all_checks_passed = True
                for method_name, skeleton in source_results.items():
                    checks = [
                        ("2D 배열", skeleton.ndim == 2),
                        ("uint8 타입", skeleton.dtype == np.uint8),
                        ("이진 값 (0 or 255)", set(np.unique(skeleton)).issubset({0, 255})),
                        ("공간 차원 유지", skeleton.shape[:2] == source_img.shape[:2]),
                    ]
                    
                    for check_name, check_result in checks:
                        if not check_result:
                            print(f"     ✗ {method_name}: {check_name}")
                            all_checks_passed = False
                            all_passed = False
                
                if all_checks_passed:
                    print(f"   ✓ Source 이미지 처리 성공 (모든 방법)")
                else:
                    print(f"   ✗ Source 이미지 처리 실패 (일부 방법)")
                
                # 저장
                if output_dir:
                    # Pipeline E 결과 저장 (비교용)
                    if source_e is not None:
                        source_e_filename = f"{pair_id}_source_pipeline_e.png"
                        cv2.imwrite(str(output_dir / source_e_filename), source_e)
                    
                    # 각 골격화 방법 결과 저장
                    saved_files = []
                    for method_name, skeleton in source_results.items():
                        filename = f"{pair_id}_source_e_sk_{method_name}.png"
                        cv2.imwrite(str(output_dir / filename), skeleton)
                        saved_files.append(filename)
                    
                    print(f"   → 저장: {', '.join(saved_files)}")
                    
            else:
                # 단일 방법만 적용
                source_sk = process(source_img, skeleton_method=skeleton_method)
                print(f"   - Source 처리 완료: {source_sk.shape}, dtype={source_sk.dtype}")
                
                # 통계 비교
                skeleton_pixels = np.sum(source_sk == 255)
                skeleton_ratio = 100 * skeleton_pixels / source_sk.size
                
                from scipy import ndimage
                _, num_components = ndimage.label(source_sk == 255)
                
                print(f"     통계:")
                if source_e is not None:
                    edge_pixels_e = np.sum(source_e == 255)
                    edge_ratio_e = 100 * edge_pixels_e / source_e.size
                    reduction = 100 * (edge_pixels_e - skeleton_pixels) / max(edge_pixels_e, 1)
                    print(f"       Pipeline E: 에지={edge_pixels_e:,} ({edge_ratio_e:.2f}%)")
                    print(f"       Pipeline E-SK: 에지={skeleton_pixels:,} ({skeleton_ratio:.2f}%), "
                          f"감소율={reduction:.1f}%")
                else:
                    print(f"       Pipeline E-SK: 에지={skeleton_pixels:,} ({skeleton_ratio:.2f}%), "
                          f"컴포넌트={num_components:,}")
                
                # 검증
                checks = [
                    ("2D 배열", source_sk.ndim == 2),
                    ("uint8 타입", source_sk.dtype == np.uint8),
                    ("이진 값 (0 or 255)", set(np.unique(source_sk)).issubset({0, 255})),
                    ("공간 차원 유지", source_sk.shape[:2] == source_img.shape[:2]),
                ]
                
                all_checks_passed = True
                for check_name, check_result in checks:
                    status = "✓" if check_result else "✗"
                    print(f"     {status} {check_name}")
                    if not check_result:
                        all_checks_passed = False
                        all_passed = False
                
                if all_checks_passed:
                    print(f"   ✓ Source 이미지 처리 성공")
                else:
                    print(f"   ✗ Source 이미지 처리 실패")
                
                # 저장
                if output_dir:
                    if source_e is not None:
                        source_e_filename = f"{pair_id}_source_pipeline_e.png"
                        cv2.imwrite(str(output_dir / source_e_filename), source_e)
                    
                    source_sk_filename = f"{pair_id}_source_e_sk_{skeleton_method}.png"
                    cv2.imwrite(str(output_dir / source_sk_filename), source_sk)
                    print(f"   → 저장: {source_sk_filename}")
                
        except Exception as e:
            print(f"   ✗ Source 이미지 처리 오류: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
        
        # Target 이미지 처리 (Source와 동일한 로직)
        try:
            target_img = load_image(str(project_root / target_path), grayscale=False)
            print(f"   - Target 로드: {target_img.shape}, dtype={target_img.dtype}")
            
            # Pipeline E-SK 처리
            if skeleton_method == "all":
                target_results = process(target_img, skeleton_method="all")
                print(f"   - Target 처리 완료: 3가지 방법")
                
                # 저장
                if output_dir:
                    saved_files = []
                    for method_name, skeleton in target_results.items():
                        filename = f"{pair_id}_target_e_sk_{method_name}.png"
                        cv2.imwrite(str(output_dir / filename), skeleton)
                        saved_files.append(filename)
                    print(f"   → 저장: {', '.join(saved_files)}")
            else:
                target_sk = process(target_img, skeleton_method=skeleton_method)
                print(f"   - Target 처리 완료: {target_sk.shape}, dtype={target_sk.dtype}")
                
                # 저장
                if output_dir:
                    target_sk_filename = f"{pair_id}_target_e_sk_{skeleton_method}.png"
                    cv2.imwrite(str(output_dir / target_sk_filename), target_sk)
                    print(f"   → 저장: {target_sk_filename}")
                
        except Exception as e:
            print(f"   ✗ Target 이미지 처리 오류: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    # 결과 요약
    print(f"\n{'=' * 70}")
    if all_passed:
        print("✓ Pipeline E-SK 검증 성공: 모든 이미지가 정상적으로 처리되었습니다.")
        if save_outputs:
            print(f"✓ 결과 저장 위치: {output_dir}")
        print("=" * 70)
        return True
    else:
        print("✗ Pipeline E-SK 검증 실패: 일부 이미지 처리에 문제가 있습니다.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline E-SK 동작 검증 스크립트")
    parser.add_argument("--no-save", action="store_true", 
                        help="처리된 이미지를 저장하지 않음")
    parser.add_argument("--skeleton-method", type=str, default="all",
                        choices=["zhang_suen", "morphological", "medial_axis", "all"],
                        help="골격화 방법 (기본값: all)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Pipeline E와 비교하지 않음")
    args = parser.parse_args()
    
    success = verify_pipeline_e_sk(
        save_outputs=not args.no_save,
        skeleton_method=args.skeleton_method,
        compare_with_pipeline_e=not args.no_compare
    )
    sys.exit(0 if success else 1)
