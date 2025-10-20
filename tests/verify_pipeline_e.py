"""
Pipeline E 동작 검증 스크립트
실제 이미지를 로드하고 Pipeline E를 적용하여 결과를 확인하고 저장합니다.
Pipeline E는 노이즈 필터링된 경계(edge) 검출 기반 전처리를 수행합니다.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hpe.data.loader import load_image_pairs_from_csv, load_image
from src.hpe.pipelines.pipeline_e import process
from src.hpe.pipelines.pipeline_d import process as pipeline_d_process


def verify_pipeline_e(
    save_outputs: bool = True,
    edge_method: str = "sobel",
    filter_method: str = "connected_components",
    min_component_size: int = 50
):
    """Pipeline E의 동작을 실제 이미지로 검증
    
    Args:
        save_outputs: True인 경우 처리된 이미지를 output 폴더에 저장
        edge_method: 에지 검출 방법 ('sobel' 또는 'canny')
        filter_method: 필터링 방법 ('connected_components' 또는 'morphology')
        min_component_size: 최소 컴포넌트 크기 (픽셀 수)
    """
    
    print("=" * 70)
    print(f"Pipeline E 동작 검증 시작")
    print(f"  edge_method={edge_method}")
    print(f"  filter_method={filter_method}")
    print(f"  min_component_size={min_component_size}")
    print("=" * 70)
    
    # Output 폴더 생성
    output_dir = None
    if save_outputs:
        output_dir = project_root / "output" / "pipeline_e"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n출력 폴더: {output_dir}")
    
    # 1. CSV 파일에서 이미지 쌍 로드
    csv_path = project_root / "tests" / "fixtures" / "test_pipeline_beta.csv"
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
    print(f"2. Pipeline E 적용 (Pipeline D 결과와 비교)")
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
            
            # Pipeline D와 Pipeline E 모두 처리
            source_d = pipeline_d_process(source_img, edge_method=edge_method)
            source_e = process(
                source_img, 
                edge_method=edge_method,
                filter_method=filter_method,
                min_component_size=min_component_size
            )
            
            print(f"   - Source 처리 완료: {source_e.shape}, dtype={source_e.dtype}")
            
            # 통계 비교
            edge_pixels_d = np.sum(source_d == 255)
            edge_pixels_e = np.sum(source_e == 255)
            total_pixels = source_e.size
            
            edge_ratio_d = 100 * edge_pixels_d / total_pixels
            edge_ratio_e = 100 * edge_pixels_e / total_pixels
            reduction_ratio = 100 * (edge_pixels_d - edge_pixels_e) / max(edge_pixels_d, 1)
            
            # Connected components 개수 비교
            from scipy import ndimage
            _, num_components_d = ndimage.label(source_d == 255)
            _, num_components_e = ndimage.label(source_e == 255)
            component_reduction = 100 * (num_components_d - num_components_e) / max(num_components_d, 1)
            
            print(f"     통계 비교:")
            print(f"       Pipeline D: 에지={edge_pixels_d:,} ({edge_ratio_d:.2f}%), "
                  f"컴포넌트={num_components_d:,}")
            print(f"       Pipeline E: 에지={edge_pixels_e:,} ({edge_ratio_e:.2f}%), "
                  f"컴포넌트={num_components_e:,}")
            print(f"       감소율: 에지 {reduction_ratio:.1f}%, 컴포넌트 {component_reduction:.1f}%")
            
            # 검증
            checks = [
                ("2D 배열", source_e.ndim == 2),
                ("uint8 타입", source_e.dtype == np.uint8),
                ("이진 값 (0 or 255)", set(np.unique(source_e)).issubset({0, 255})),
                ("공간 차원 유지", source_e.shape[:2] == source_img.shape[:2]),
                ("에지 비율 적절", 0 <= edge_ratio_e < 25),
                ("노이즈 감소", edge_pixels_e <= edge_pixels_d),
                ("컴포넌트 감소", num_components_e <= num_components_d),
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
                # Pipeline D 결과 저장
                source_d_filename = f"{pair_id}_source_pipeline_d.png"
                source_d_path = output_dir / source_d_filename
                cv2.imwrite(str(source_d_path), source_d)
                
                # Pipeline E 결과 저장
                source_e_filename = f"{pair_id}_source_pipeline_e.png"
                source_e_path = output_dir / source_e_filename
                cv2.imwrite(str(source_e_path), source_e)
                
                print(f"   → 저장: {source_d_filename}, {source_e_filename}")
                
        except Exception as e:
            print(f"   ✗ Source 이미지 처리 오류: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
        
        # Target 이미지 처리
        try:
            target_img = load_image(str(project_root / target_path), grayscale=False)
            print(f"   - Target 로드: {target_img.shape}, dtype={target_img.dtype}")
            
            # Pipeline D와 Pipeline E 모두 처리
            target_d = pipeline_d_process(target_img, edge_method=edge_method)
            target_e = process(
                target_img,
                edge_method=edge_method,
                filter_method=filter_method,
                min_component_size=min_component_size
            )
            
            print(f"   - Target 처리 완료: {target_e.shape}, dtype={target_e.dtype}")
            
            # 통계 비교
            edge_pixels_d = np.sum(target_d == 255)
            edge_pixels_e = np.sum(target_e == 255)
            total_pixels = target_e.size
            
            edge_ratio_d = 100 * edge_pixels_d / total_pixels
            edge_ratio_e = 100 * edge_pixels_e / total_pixels
            reduction_ratio = 100 * (edge_pixels_d - edge_pixels_e) / max(edge_pixels_d, 1)
            
            # Connected components 개수 비교
            from scipy import ndimage
            _, num_components_d = ndimage.label(target_d == 255)
            _, num_components_e = ndimage.label(target_e == 255)
            component_reduction = 100 * (num_components_d - num_components_e) / max(num_components_d, 1)
            
            print(f"     통계 비교:")
            print(f"       Pipeline D: 에지={edge_pixels_d:,} ({edge_ratio_d:.2f}%), "
                  f"컴포넌트={num_components_d:,}")
            print(f"       Pipeline E: 에지={edge_pixels_e:,} ({edge_ratio_e:.2f}%), "
                  f"컴포넌트={num_components_e:,}")
            print(f"       감소율: 에지 {reduction_ratio:.1f}%, 컴포넌트 {component_reduction:.1f}%")
            
            # 검증
            checks = [
                ("2D 배열", target_e.ndim == 2),
                ("uint8 타입", target_e.dtype == np.uint8),
                ("이진 값 (0 or 255)", set(np.unique(target_e)).issubset({0, 255})),
                ("공간 차원 유지", target_e.shape[:2] == target_img.shape[:2]),
                ("에지 비율 적절", 0 <= edge_ratio_e < 25),
                ("노이즈 감소", edge_pixels_e <= edge_pixels_d),
                ("컴포넌트 감소", num_components_e <= num_components_d),
            ]
            
            all_checks_passed = True
            for check_name, check_result in checks:
                status = "✓" if check_result else "✗"
                print(f"     {status} {check_name}")
                if not check_result:
                    all_checks_passed = False
                    all_passed = False
            
            if all_checks_passed:
                print(f"   ✓ Target 이미지 처리 성공")
            else:
                print(f"   ✗ Target 이미지 처리 실패")
            
            # 저장
            if output_dir:
                # Pipeline D 결과 저장
                target_d_filename = f"{pair_id}_target_pipeline_d.png"
                target_d_path = output_dir / target_d_filename
                cv2.imwrite(str(target_d_path), target_d)
                
                # Pipeline E 결과 저장
                target_e_filename = f"{pair_id}_target_pipeline_e.png"
                target_e_path = output_dir / target_e_filename
                cv2.imwrite(str(target_e_path), target_e)
                
                print(f"   → 저장: {target_d_filename}, {target_e_filename}")
                
        except Exception as e:
            print(f"   ✗ Target 이미지 처리 오류: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    
    # 결과 요약
    print(f"\n{'=' * 70}")
    if all_passed:
        print("✓ Pipeline E 검증 성공: 모든 이미지가 정상적으로 처리되었습니다.")
        if save_outputs:
            print(f"✓ 결과 저장 위치: {output_dir}")
        print("=" * 70)
        return True
    else:
        print("✗ Pipeline E 검증 실패: 일부 이미지 처리에 문제가 있습니다.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline E 동작 검증 스크립트")
    parser.add_argument("--no-save", action="store_true", 
                        help="처리된 이미지를 저장하지 않음")
    parser.add_argument("--edge-method", type=str, default="sobel",
                        choices=["sobel", "canny"],
                        help="에지 검출 방법 (기본값: sobel)")
    parser.add_argument("--filter-method", type=str, default="connected_components",
                        choices=["connected_components", "morphology"],
                        help="노이즈 필터링 방법 (기본값: connected_components)")
    parser.add_argument("--min-component-size", type=int, default=50,
                        help="최소 컴포넌트 크기 픽셀 수 (기본값: 50)")
    args = parser.parse_args()
    
    success = verify_pipeline_e(
        save_outputs=not args.no_save,
        edge_method=args.edge_method,
        filter_method=args.filter_method,
        min_component_size=args.min_component_size
    )
    sys.exit(0 if success else 1)
