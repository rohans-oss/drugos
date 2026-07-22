import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

// In test mode we always create a fresh client so each test process picks
// up the test DATABASE_URL set by tests/api/setup.ts.
const shouldUseGlobalCache = process.env.NODE_ENV !== 'production' && process.env.NODE_ENV !== 'test'

export const db =
  (shouldUseGlobalCache ? globalForPrisma.prisma : undefined) ??
  new PrismaClient(
    process.env.NODE_ENV === 'test'
      ? { datasources: { db: { url: process.env.DATABASE_URL } } }
      : { log: ['query'] }
  )

if (shouldUseGlobalCache) globalForPrisma.prisma = db