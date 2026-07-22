import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()
async function check() {
  const users = await prisma.user.findMany({ select: { email: true, emailVerified: true, status: true } });
  console.log(JSON.stringify(users, null, 2));
}
check().finally(() => prisma.$disconnect());
