# verify_wish_machine.py

import asyncio
from playwright.async_api import async_playwright
import subprocess
import os
import signal

async def main():
    # Start the Streamlit server as a background process
    command = ["streamlit", "run", "src/app/streamlit_app.py", "--server.port", "8502"]
    with open("streamlit_server.log", "w") as log_file:
        server_process = subprocess.Popen(command, stdout=log_file, stderr=log_file, preexec_fn=os.setsid)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            # Give the server a moment to start
            await asyncio.sleep(10) # Increased wait time for server startup

            # Go to the Streamlit app
            await page.goto("http://localhost:8502")

            # Wait for the app to load and take a screenshot
            await page.wait_for_selector("text=Chatbot", timeout=60000)
            await page.screenshot(path="wish_machine_before_click.png")

            # Click the "Wish Machine" tab
            await page.get_by_text("Wish Machine").click()

            # Wait for the page to load
            await page.wait_for_selector("text=Device Connection", timeout=60000)

            # Take a screenshot
            await page.screenshot(path="wish_machine_after_click.png")

            print("Successfully verified the Wish Machine tab.")

        except Exception as e:
            print(f"An error occurred: {e}")
            await page.screenshot(path="wish_machine_error.png")

        finally:
            await browser.close()
            # Terminate the server process group
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)

if __name__ == "__main__":
    asyncio.run(main())
