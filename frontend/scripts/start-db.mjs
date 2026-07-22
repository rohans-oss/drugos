/**
 * start-db.mjs — Starts an embedded PostgreSQL server for local development.
 * No Docker, no system install, no admin rights needed.
 *
 * Usage: node scripts/start-db.mjs
 *
 * Then in another terminal run: npm run dev
 */

import EmbeddedPostgres from 'embedded-postgres';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync, rmSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const dataDir = join(__dirname, '..', '.postgres-data');

const pg = new EmbeddedPostgres({
  databaseDir: dataDir,
  user: 'postgres',
  password: 'postgres',
  port: 5432,
  persistent: true,   // data survives restarts
});

console.log('🐘 Starting embedded PostgreSQL on port 5432...');

try {
  // Remove stale postmaster.pid if present from previous ungraceful shutdown
  const pidFile = join(dataDir, 'postmaster.pid');
  if (existsSync(pidFile)) {
    try { rmSync(pidFile, { force: true }); } catch {}
  }

  const isInitialized = existsSync(join(dataDir, 'PG_VERSION'));
  if (!isInitialized) {
    await pg.initialise();
  }
  await pg.start();

  // Create the drugos database if it doesn't exist
  try {
    await pg.createDatabase('drugos');
    console.log('✅ Created database: drugos');
  } catch (e) {
    if (e.message?.includes('already exists')) {
      console.log('ℹ️  Database "drugos" already exists — skipping create.');
    } else {
      console.error('⚠️  Could not create database:', e.message);
    }
  }

  console.log('');
  console.log('✅ PostgreSQL is running!');
  console.log('   Host: localhost');
  console.log('   Port: 5432');
  console.log('   User: postgres');
  console.log('   Password: postgres');
  console.log('   Database: drugos');
  console.log('');
  console.log('👉 Now open a SECOND terminal and run:');
  console.log('   cd frontend');
  console.log('   npx prisma migrate dev --name init');
  console.log('   npm run dev');
  console.log('');
  console.log('Keep this terminal open while developing!');
  console.log('Press Ctrl+C to stop the database.\n');

  // Keep running until Ctrl+C
  process.on('SIGINT', async () => {
    console.log('\n🛑 Stopping PostgreSQL...');
    await pg.stop();
    process.exit(0);
  });

} catch (err) {
  console.error('❌ Failed to start embedded PostgreSQL:', err);
  process.exit(1);
}
