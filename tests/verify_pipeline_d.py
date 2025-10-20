"""
Pipeline D 동작 검증 스크립트
실제 이미지를 로드하고 Pipeline D를 적용하여 결과를 확인하고 저장합니다.
Pipeline D는 경계(edge) 검출 기반 전처리를 수행합니다.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hpe.data.loader import load_image_pairs_from_csv, load_image
from src.hpe.pipelines.pipeline_d import process


def process_with_intermediates(image: np.ndarray):
    """Pipeline D를 단계별 중간 결과와 함께 처리
    
    Returns:
        dict: 각 단계별 결과 이미지를 담은 딕셔너리
    """
    from skimage.feature import canny
    
    results = {}
    
    # Step 1: 그레이스케일 변환
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    results['1_grayscale'] = gray
    
    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=1.2)
    results['2_blurred'] = blurred
    
    # Step 3: 임계값 계산 (uint8 범위에서 median 계산 후 정규화)
    median_uint8 = float(np.median(blurred))
    median_val = median_uint8 / 255.0
    low_threshold = np.clip(0.66 * median_val, 0.05, 0.3)
    high_threshold = np.clip(1.33 * median_val, low_threshold + 0.05, 0.5)
    
    # Canny를 위한 정규화
    gray_norm = blurred.astype(np.float32) / 255.0
    
    results['threshold_info'] = {
        'median': median_val,
        'low': low_threshold,
        'high': high_threshold
    }
    
    # Step 4: Canny 에지 검출
    edges = canny(
        gray_norm,
        sigma=1.5,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    results['3_edges_raw'] = (edges * 255).astype(np.uint8)
    
    # Step 5: 배경 억제 마스크
    background_mask = gray > 245
    results['4_background_mask'] = (background_mask * 255).astype(np.uint8)
    
    # Step 6: 배경 억제 적용
    edges_filtered = edges & (~background_mask)
    results['5_edges_filtered'] = (edges_filtered * 255).astype(np.uint8)
    
    return results


def verify_pipeline_d(
    save_outputs: bool = True, 
    save_intermediates: bool = False,
    edge_method: str = "sobel"
):
    """Pipeline D의 동작을 실제 이미지로 검증
    
    Args:
        save_outputs: True인 경우 처리된 이미지를 output 폴더에 저장
        save_intermediates: True인 경우 단계별 중간 결과도 저장
        edge_method: 에지 검출 방법 ('sobel' 또는 'canny')
    """
    
    print("=" * 70)
    print(f"Pipeline D 동작 검증 시작 (edge_method={edge_method})")
    print("=" * 70)
    
    # Output 폴더 생성
    output_dir = None
    intermediates_dir = None
    if save_outputs:
        output_dir = project_root / "output" / "pipeline_d"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n출력 폴더: {output_dir}")
        
        if save_intermediates:
            intermediates_dir = output_dir / "intermediates"
            intermediates_dir.mkdir(parents=True, exist_ok=True)
            print(f"중간 단계 폴더: {intermediates_dir}")
    
    # 1. CSV 파일에서 이미지 쌍 로드
    csv_path = project_root / "tests" / "fixtures" / "test_pipeline_a.csv"
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
    print(f"2. Pipeline D 적용")
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
            source_img = load_image(str(project_root / source_path))
            print(f"   - Source 로드: {source_img.shape}, dtype={source_img.dtype}")
            
            # 중간 단계 저장 여부에 따라 처리
            if save_intermediates:
                intermediates = process_with_intermediates(source_img)
                source_processed = intermediates['5_edges_filtered']
                
                # 중간 단계 저장
                if intermediates_dir:
                    for step_name, step_img in intermediates.items():
                        if step_name == 'threshold_info':
                            continue
                        step_filename = f"{pair_id}_source_{step_name}.png"
                        step_path = intermediates_dir / step_filename
                        cv2.imwrite(str(step_path), step_img)
                    
                    # 임계값 정보 저장
                    info = intermediates['threshold_info']
                    info_filename = f"{pair_id}_source_thresholds.txt"
                    info_path = intermediates_dir / info_filename
                    with open(info_path, 'w') as f:
                        f.write(f"Median: {info['median']:.4f}\n")
                        f.write(f"Low threshold: {info['low']:.4f}\n")
                        f.write(f"High threshold: {info['high']:.4f}\n")
                    
                    print(f"     → 중간 단계 저장 완료: {len(intermediates)-1}개 파일")
            else:
                source_processed = process(source_img, edge_method=edge_method)
            
            print(f"   - Source 처리 완료: {source_processed.shape}, dtype={source_processed.dtype}")
            
            # 에지 통계
            edge_pixels = np.sum(source_processed == 255)
            total_pixels = source_processed.size
            edge_ratio = edge_pixels / total_pixels
            
            print(f"     통계: 에지 픽셀={edge_pixels:,} ({edge_ratio:.2%}), "
                  f"배경 픽셀={total_pixels - edge_pixels:,}")
            
            # 검증
            checks = [
                ("2D 배열", source_processed.ndim == 2),
                ("uint8 타입", source_processed.dtype == np.uint8),
                ("이진 값 (0 or 255)", set(np.unique(source_processed)).issubset({0, 255})),
                ("공간 차원 유지", source_processed.shape[:2] == source_img.shape[:2]),
                ("에지 비율 적절", 0 < edge_ratio < 0.3),  # 0~30% 범위
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
                
                # 처리된 이미지 저장
                if save_outputs and output_dir is not None:
                    output_filename = f"{pair_id}_source.png"
                    output_path = output_dir / output_filename
                    cv2.imwrite(str(output_path), source_processed)
                    print(f"   → 저장: {output_filename}")
            else:
                print(f"   ✗ Source 이미지 처리 실패")
                
        except Exception as e:
            print(f"   ✗ Source 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            continue
        
        # Target 이미지 처리
        try:
            target_img = load_image(str(project_root / target_path))
            print(f"   - Target 로드: {target_img.shape}, dtype={target_img.dtype}")
            
            # 중간 단계 저장 여부에 따라 처리
            if save_intermediates:
                intermediates = process_with_intermediates(target_img)
                target_processed = intermediates['5_edges_filtered']
                
                # 중간 단계 저장
                if intermediates_dir:
                    for step_name, step_img in intermediates.items():
                        if step_name == 'threshold_info':
                            continue
                        step_filename = f"{pair_id}_target_{step_name}.png"
                        step_path = intermediates_dir / step_filename
                        cv2.imwrite(str(step_path), step_img)
                    
                    # 임계값 정보 저장
                    info = intermediates['threshold_info']
                    info_filename = f"{pair_id}_target_thresholds.txt"
                    info_path = intermediates_dir / info_filename
                    with open(info_path, 'w') as f:
                        f.write(f"Median: {info['median']:.4f}\n")
                        f.write(f"Low threshold: {info['low']:.4f}\n")
                        f.write(f"High threshold: {info['high']:.4f}\n")
                    
                    print(f"     → 중간 단계 저장 완료: {len(intermediates)-1}개 파일")
            else:
                target_processed = process(target_img, edge_method=edge_method)
            
            print(f"   - Target 처리 완료: {target_processed.shape}, dtype={target_processed.dtype}")
            
            # 에지 통계
            edge_pixels = np.sum(target_processed == 255)
            total_pixels = target_processed.size
            edge_ratio = edge_pixels / total_pixels
            
            print(f"     통계: 에지 픽셀={edge_pixels:,} ({edge_ratio:.2%}), "
                  f"배경 픽셀={total_pixels - edge_pixels:,}")
            
            # 검증
            checks = [
                ("2D 배열", target_processed.ndim == 2),
                ("uint8 타입", target_processed.dtype == np.uint8),
                ("이진 값 (0 or 255)", set(np.unique(target_processed)).issubset({0, 255})),
                ("공간 차원 유지", target_processed.shape[:2] == target_img.shape[:2]),
                ("에지 비율 적절", 0 < edge_ratio < 0.3),
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
                
                # 처리된 이미지 저장
                if save_outputs and output_dir is not None:
                    output_filename = f"{pair_id}_target.png"
                    output_path = output_dir / output_filename
                    cv2.imwrite(str(output_path), target_processed)
                    print(f"   → 저장: {output_filename}")
            else:
                print(f"   ✗ Target 이미지 처리 실패")
                
        except Exception as e:
            print(f"   ✗ Target 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 3. 최종 결과
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ Pipeline D 검증 성공: 모든 이미지가 정상적으로 처리되었습니다.")
        if save_outputs:
            print(f"✓ 결과 저장 위치: {output_dir}")
        print("=" * 70)
        return True
    else:
        print("✗ Pipeline D 검증 실패: 일부 이미지 처리에 문제가 있습니다.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline D 동작 검증 스크립트")
    parser.add_argument("--no-save", action="store_true", 
                        help="처리된 이미지를 저장하지 않음")
    parser.add_argument("--intermediates", action="store_true",
                        help="단계별 중간 결과 이미지 저장")
    parser.add_argument("--edge-method", type=str, default="sobel",
                        choices=["sobel", "canny"],
                        help="에지 검출 방법 (기본값: sobel)")
    args = parser.parse_args()
    
    success = verify_pipeline_d(
        save_outputs=not args.no_save,
        save_intermediates=args.intermediates,
        edge_method=args.edge_method
    )
    sys.exit(0 if success else 1)
