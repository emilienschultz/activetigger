import { useCallback } from 'react';

import { NotificationData } from '../types';
import { useAppContext } from './context';

let INCREMENTAL_ID = 1;
export function useNotifications() {
  const { setAppContext } = useAppContext();

  const notify = useCallback(
    (notif: NotificationData) => {
      const id = ++INCREMENTAL_ID;
      setAppContext(
        // in a setter we can use a state modification method
        (state) =>
          // the param is the current state
          ({
            ...state, // here we want to keep the current state object untouched so we spread it
            // but we update the notifications key with it's new value
            notifications: [{ id, createdAt: new Date(), ...notif }, ...state.notifications],
          }),
      );
    },
    [setAppContext],
  );

  return { notify };
}
