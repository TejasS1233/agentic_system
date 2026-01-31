class Dispatcher:
    def __init__(self):
        pass

    def route(self, task_description: str, file_context: list[str] = []) -> str:
        """
        Determines if a task should go to the Public (Cloud) or Private (Local) zone.
        Returns: 'public' | 'private'
        """
        # Rule 1: Keyworks
        private_keywords = [
            "secret",
            "key",
            "password",
            "private",
            ".env",
            "sensitive",
            "bearer",
            "token",
        ]
        if any(k in task_description.lower() for k in private_keywords):
            print("[Dispatcher] Sensitive keyword detected -> Routing to PRIVATE ZONE")
            return "private"

        # Rule 2: File context
        for f in file_context:
            if ".env" in f or "id_rsa" in f:
                print(
                    f"[Dispatcher] Sensitive file '{f}' in context -> Routing to PRIVATE ZONE"
                )
                return "private"

        print(
            "[Dispatcher] No sensitivity detected -> Routing to PUBLIC ZONE (Cheaper/Faster)"
        )
        return "public"
