import json
import argparse
from pathlib import Path

def generate_markdown(instance_id, real_traj_path, output_md_path):
    if not real_traj_path.exists():
        print(f"Error: Real trajectory not found at {real_traj_path}")
        return

    with open(real_traj_path, "r") as f:
        data = json.load(f)
    
    history = data.get("history", [])
    
    md_content = [
        f"# Semantic Mapping Comparison: {instance_id}",
        f"\n**Trajectory Source:** `{real_traj_path}`",
        "\n---"
    ]

    step_count = 1
    for i in range(len(history)):
        item = history[i]
        role = item.get("role")
        
        if role == "assistant":
            md_content.append(f"\n## 🔢 Step {step_count}")
            md_content.append("### 🤖 Assistant Action")
            
            virtual_action = item.get("virtual_content", "N/A")
            real_action = item.get("content", "N/A")
            
            md_content.append("| Type | Content |")
            md_content.append("| :--- | :--- |")
            md_content.append(f"| **Virtual (Agent's Mind)** | `{virtual_action}` |")
            md_content.append(f"| **Real (Docker Executed)** | `{real_action}` |")
            
            if "```" in virtual_action or len(virtual_action) > 100:
                md_content.append("\n**Virtual Detail:**")
                md_content.append(f"```bash\n{virtual_action}\n```")
                md_content.append("**Real Detail:**")
                md_content.append(f"```bash\n{real_action}\n```")
            
        elif role == "user":
            md_content.append("\n### 📥 Environment Observation")
            
            virtual_obs = item.get("virtual_content", "N/A")
            real_obs = item.get("content", "N/A")
            
            md_content.append("<details>")
            md_content.append("<summary><b>View Observation Comparison</b> (Click to expand)</summary>\n")
            
            md_content.append("#### 🟢 Virtual Observation (What Agent sees)")
            md_content.append(f"```text\n{virtual_obs}\n```")
            
            md_content.append("#### 🔵 Real Observation (Actual data)")
            md_content.append(f"```text\n{real_obs}\n```")
            
            md_content.append("</details>")
            md_content.append("\n---")
            step_count += 1

    with open(output_md_path, "w") as f:
        f.write("\n".join(md_content))
    
    print(f"✨ Comparison Markdown generated: {output_md_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare virtual and real trajectories.")
    parser.add_argument("--instance-id", required=True, help="Instance ID to compare")
    parser.add_argument("--run-dir", type=Path, required=True, help="The output directory of the run (e.g. output/django_mapped_test1)")
    
    args = parser.parse_args()
    
    instance_path = args.run_dir / args.instance_id
    real_traj = instance_path / f"{args.instance_id}.real.traj.json"
    output_md = instance_path / f"{args.instance_id}_comparison.md"
    
    generate_markdown(args.instance_id, real_traj, output_md)

if __name__ == "__main__":
    main()
