"""默认Dry Run地匹配文献；--apply才会保留原件并创建规范副本。"""
import argparse
from pathlib import Path
from knowledge_base.source_catalog import apply_operations, atomic_write_json, build_match_report, write_match_reports, write_source_register

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="knowledge_sources")
    parser.add_argument("--spreadsheet", default="knowledge_sources/RAPPO_第五天_知识库文献清单.xlsx")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        records, matches, reconciliation = build_match_report(args.root, args.spreadsheet)
    except ValueError as error:
        atomic_write_json(Path(args.root) / "reports" / "selection_reconciliation.json", {
            "pdf_file_count": len(list(Path(args.root).glob("*.pdf"))), "spreadsheet_row_count": 0,
            "spreadsheet_a_core_count": 0, "spreadsheet_include_yes_count": 0,
            "matched_file_count": 0, "unmatched_files": [], "missing_expected_documents": [],
            "duplicate_matches": [], "notes": ["Excel校验失败：{0}".format(error)],
        })
        raise
    write_match_reports(args.root, matches, reconciliation)
    unresolved = [item for item in matches if item["status"] != "matched"]
    if args.apply:
        if unresolved or reconciliation["duplicate_matches"]:
            raise RuntimeError("存在未唯一匹配PDF，拒绝执行Apply")
        apply_operations(args.root, matches)
        write_source_register(args.root, records, matches)
    print({"pdf_file_count": reconciliation["pdf_file_count"], "matched_file_count": reconciliation["matched_file_count"], "unresolved": len(unresolved), "applied": args.apply})

if __name__ == "__main__":
    main()
