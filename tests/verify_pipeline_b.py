"""
Pipeline B 동작 검증 스크립트
실제 이미지를 로드하고 Pipeline B를 적용하여 결과를 확인하고 저장합니다.
Pipeline B는 헤마톡실린 채널 추출 및 선택적 LoG 필터를 지원합니다.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hpe.data.loader import load_image_pairs_from_csv, load_image
from src.hpe.pipelines.pipeline_b import process


def verify_pipeline_b(save_outputs: bool = True, test_log: bool = True):
    """Pipeline B의 동작을 실제 이미지로 검증
    
    Args:
        save_outputs: True인 경우 처리된 이미지를 output 폴더에 저장
        test_log: True인 경우 LoG 필터 적용 버전도 함께 테스트
    """
    
    print("=" * 70)
    print("Pipeline B 동작 검증 시작")
    print("=" * 70)
    
    # Output 폴더 생성 (기본 모드와 LoG 모드)
    modes = [("basic", False)]
    if test_log:
        modes.append(("clog", True))
    
    output_dirs = {}
    if save_outputs:
        for mode_name, _ in modes:
            output_dir = project_root / "output" / f"pipeline_b_{mode_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[mode_name] = output_dir
            print(f"\n출력 폴더 ({mode_name}): {output_dir}")
    
    # 1. CSV 파일에서 이미지 쌍 로드
    #csv_path = project_root / "tests" / "fixtures" / "test_pipeline_a.csv"
    csv_path = project_root / "tests" / "fixtures" / "test_pipeline.csv"
    print(f"\n1. CSV 파일 로드: {csv_path}")
    
    try:
        pairs = load_image_pairs_from_csv(str(csv_path))
        print(f"   ✓ {len(pairs)}개의 이미지 쌍을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"   ✗ CSV 로드 실패: {e}")
        return False
    
    # 2. 각 모드에 대해 이미지 쌍 처리
    all_passed = True
    
    for mode_name, use_clog in modes:
        print(f"\n{'=' * 70}")
        print(f"2. Pipeline B 적용 (모드: {mode_name.upper()}, use_clog={use_clog})")
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
                
                source_processed = process(source_img, use_clog=use_clog)
                print(f"   - Source 처리 완료: {source_processed.shape}, dtype={source_processed.dtype}")
                print(f"     통계: min={source_processed.min()}, max={source_processed.max()}, "
                      f"mean={source_processed.mean():.2f}, std={source_processed.std():.2f}")
                
                # 검증
                checks = [
                    ("2D 배열", source_processed.ndim == 2),
                    ("uint8 타입", source_processed.dtype == np.uint8),
                    ("0-255 범위", source_processed.min() >= 0 and source_processed.max() <= 255),
                    ("공간 차원 유지", source_processed.shape[:2] == source_img.shape[:2]),
                    ("비어있지 않음", source_processed.sum() > 0),
                    ("분산 있음", np.var(source_processed) > 0),
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
                    if save_outputs:
                        output_filename = f"{pair_id}_source_{mode_name}.png"
                        output_path = output_dirs[mode_name] / output_filename
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
                
                target_processed = process(target_img, use_clog=use_clog)
                print(f"   - Target 처리 완료: {target_processed.shape}, dtype={target_processed.dtype}")
                print(f"     통계: min={target_processed.min()}, max={target_processed.max()}, "
                      f"mean={target_processed.mean():.2f}, std={target_processed.std():.2f}")
                
                # 검증
                checks = [
                    ("2D 배열", target_processed.ndim == 2),
                    ("uint8 타입", target_processed.dtype == np.uint8),
                    ("0-255 범위", target_processed.min() >= 0 and target_processed.max() <= 255),
                    ("공간 차원 유지", target_processed.shape[:2] == target_img.shape[:2]),
                    ("비어있지 않음", target_processed.sum() > 0),
                    ("분산 있음", np.var(target_processed) > 0),
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
                    if save_outputs:
                        output_filename = f"{pair_id}_target_{mode_name}.png"
                        output_path = output_dirs[mode_name] / output_filename
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
        print("✓ Pipeline B 검증 성공: 모든 이미지가 정상적으로 처리되었습니다.")
        if save_outputs:
            for mode_name, _ in modes:
                print(f"✓ {mode_name.upper()} 모드 결과: {output_dirs[mode_name]}")
        print("=" * 70)
        return True
    else:
        print("✗ Pipeline B 검증 실패: 일부 이미지 처리에 문제가 있습니다.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline B 동작 검증 스크립트")
    parser.add_argument("--no-save", action="store_true", 
                        help="처리된 이미지를 저장하지 않음")
    parser.add_argument("--no-log", action="store_true",
                        help="LoG 필터 버전을 테스트하지 않음 (기본 모드만)")
    args = parser.parse_args()
    
    success = verify_pipeline_b(
        save_outputs=not args.no_save,
        test_log=not args.no_log
    )
    sys.exit(0 if success else 1)
