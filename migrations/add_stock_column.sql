-- Run this in Supabase SQL Editor (Dashboard > SQL Editor)
-- Adds stock column to all 8 product tables

ALTER TABLE "CPU" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "GPU" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "Motherboard" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "Case" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "Memory" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "PSU" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "CPU Cooler" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;
ALTER TABLE "Internal Drive" ADD COLUMN IF NOT EXISTS stock INTEGER NOT NULL DEFAULT 0;

-- Optional: add RLS policy allowing admins to update stock
-- (adjust the auth.uid() check to match your admin users table)
-- CREATE POLICY "admin_update_stock" ON "CPU" FOR UPDATE USING (auth.role() = 'service_role');
