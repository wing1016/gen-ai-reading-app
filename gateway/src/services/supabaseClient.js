// Supabase client for gateway
import { createClient } from '@supabase/supabase-js';

const supabaseKey =
  process.env.SUPABASE_SECRET_KEY ||
  process.env.SUPABASE_SERVICE_ROLE_KEY ||
  process.env.SUPABASE_KEY;

const supabase = createClient(
  process.env.SUPABASE_URL,
  supabaseKey
);

export default supabase;
