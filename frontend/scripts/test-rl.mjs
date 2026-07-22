async function test() {
  const login = await fetch("http://localhost:3000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: "rohan901991@gmail.com", password: "some_password" }) // We don't know the password...
  });
  // Actually, wait! The simplest is to just view the Next JS log or the Python service log.
}
