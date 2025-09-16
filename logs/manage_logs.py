import argparse
import json
from data_logging import data_logger

log = data_logger()

def list_all():
    logs = log.list_all_logs()
    print(f"\n📑 Found {len(logs)} RFP logs\n")
    for rfp_id, entry in logs.items():
        print(f"ID: {rfp_id[:12]}...")
        print(f"  Document: {entry.get('document_name')}")
        print(f"  RFP Number: {entry.get('rfp_number')}")
        print(f"  Client: {entry.get('client_name')}")
        print(f"  Last Updated: {entry.get('last_updated')}")
        print(f"  Tools Logged: {', '.join(entry.get('tools', {}).keys())}")
        print("")

def view_rfp(rfp_id):
    entry = log.get_rfp_data(rfp_id)
    if not entry:
        print(f"❌ No log found for RFP ID: {rfp_id}")
        return
    print(json.dumps(entry, indent=4))

def export_all(outfile):
    logs = log.list_all_logs()
    with open(outfile, "w") as f:
        json.dump(logs, f, indent=4)
    print(f"✅ Exported {len(logs)} logs to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage MCP RFP logs")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all logged RFPs")

    view_parser = subparsers.add_parser("view", help="View a specific RFP log")
    view_parser.add_argument("rfp_id", help="The RFP ID to view")

    export_parser = subparsers.add_parser("export", help="Export all logs to a JSON file")
    export_parser.add_argument("outfile", help="Output file path")

    args = parser.parse_args()

    if args.command == "list":
        list_all()
    elif args.command == "view":
        view_rfp(args.rfp_id)
    elif args.command == "export":
        export_all(args.outfile)
    else:
        parser.print_help()
