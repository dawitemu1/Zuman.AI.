import { exec } from "child_process";

const url = "http://localhost:3000";
let command;

if (process.platform === "win32") {
  command = `start "" "${url}"`;
} else if (process.platform === "darwin") {
  command = `open "${url}"`;
} else {
  command = `xdg-open "${url}"`;
}

exec(command, (error) => {
  if (error) {
    console.error("Failed to open browser:", error.message);
  }
});
