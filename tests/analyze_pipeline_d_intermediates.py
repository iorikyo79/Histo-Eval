#!/usr/bin/env python3
"""
Pipeline D 중간 단계 분석 스크립트
각 단계별로 에지 픽셀 수를 비교하여 어디서 에지가 손실되는지 확인
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_intermediates(intermediates_dir: str, pair_name: str, image_type: str):
    """중간 단계 이미지들을 분석하여 에지 손실을 확인"""
    
    base_path = Path(intermediates_dir)
    prefix = f"{pair_name}_{image_type}"
    
    # 각 단계별 파일 경로
    grayscale_path = base_path / f"{prefix}_1_grayscale.png"
    blurred_path = base_path / f"{prefix}_2_blurred.png"
    edges_raw_path = base_path / f"{prefix}_3_edges_raw.png"
    bg_mask_path = base_path / f"{prefix}_4_background_mask.png"
    edges_filtered_path = base_path / f"{prefix}_5_edges_filtered.png"
    thresholds_path = base_path / f"{prefix}_thresholds.txt"
    
    print(f"\n{'='*70}")
    print(f"{pair_name} - {image_type}")
    print(f"{'='*70}")
    
    # Thresholds 읽기
    with open(thresholds_path, 'r') as f:
        content = f.read()
        print(content)
    
    # 각 단계별 이미지 로드 및 분석
    grayscale = cv2.imread(str(grayscale_path), cv2.IMREAD_GRAYSCALE)
    edges_raw = cv2.imread(str(edges_raw_path), cv2.IMREAD_GRAYSCALE)
    bg_mask = cv2.imread(str(bg_mask_path), cv2.IMREAD_GRAYSCALE)
    edges_filtered = cv2.imread(str(edges_filtered_path), cv2.IMREAD_GRAYSCALE)
    
    total_pixels = grayscale.size
    
    # 통계 계산
    edge_raw_count = np.sum(edges_raw > 0)
    bg_count = np.sum(bg_mask > 0)
    edge_filtered_count = np.sum(edges_filtered > 0)
    
    edge_raw_ratio = 100 * edge_raw_count / total_pixels
    bg_ratio = 100 * bg_count / total_pixels
    edge_filtered_ratio = 100 * edge_filtered_count / total_pixels
    
    # Grayscale 통계
    gray_mean = np.mean(grayscale)
    gray_median = np.median(grayscale)
    gray_std = np.std(grayscale)
    
    print(f"이미지 크기: {grayscale.shape}, 총 픽셀: {total_pixels:,}")
    print(f"\nGrayscale 통계:")
    print(f"  평균: {gray_mean:.1f}, 중앙값: {gray_median:.1f}, 표준편차: {gray_std:.1f}")
    
    print(f"\n단계별 에지 검출:")
    print(f"  3. Canny 원본 에지:     {edge_raw_count:7,} ({edge_raw_ratio:5.2f}%)")
    print(f"  4. 배경 마스크 (>245):  {bg_count:7,} ({bg_ratio:5.2f}%)")
    print(f"  5. 필터링된 에지:       {edge_filtered_count:7,} ({edge_filtered_ratio:5.2f}%)")
    
    # 손실율 계산
    if edge_raw_count > 0:
        loss_ratio = 100 * (edge_raw_count - edge_filtered_count) / edge_raw_count
        print(f"\n에지 손실율: {loss_ratio:.1f}% (배경 필터링으로 제거)")
    else:
        print(f"\n⚠️  원본 에지가 전혀 검출되지 않았습니다!")
    
    # 문제 진단
    print(f"\n진단:")
    if edge_raw_ratio < 0.5:
        print(f"  ⚠️  Canny 에지 검출량이 매우 적습니다 ({edge_raw_ratio:.2f}%)")
        print(f"     → threshold 값이 너무 높을 가능성")
    if bg_ratio > 80:
        print(f"  ⚠️  대부분이 배경으로 판정되었습니다 ({bg_ratio:.1f}%)")
        print(f"     → 배경 threshold (>245)가 너무 낮을 가능성")
    if edge_raw_count > 0 and edge_filtered_count < edge_raw_count * 0.2:
        print(f"  ⚠️  배경 필터링으로 80% 이상의 에지가 제거되었습니다")
        print(f"     → 조직 영역의 밝기가 높아서 에지가 배경으로 오인됨")


if __name__ == "__main__":
    intermediates_dir = "/mnt/Disk1/source/Histo-Eval/output/pipeline_d/intermediates"
    
    # 모든 pair 분석
    pairs = ["pair_001", "pair_002", "pair_003", "pair_004"]
    
    for pair in pairs:
        analyze_intermediates(intermediates_dir, pair, "source")
        analyze_intermediates(intermediates_dir, pair, "target")
    
    print(f"\n{'='*70}")
    print("분석 완료")
    print(f"{'='*70}\n")
