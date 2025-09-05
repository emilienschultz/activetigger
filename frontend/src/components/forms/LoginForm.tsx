import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';

import { useGetServer, useSendResetMail } from '../../core/api';
import { useAuth } from '../../core/auth';
import { useNotifications } from '../../core/notifications';
import { LoginParams } from '../../types';

/**
 * LoginForm
 * @param redirectTo a location where the user must be redirected after successful login
 * @returns a HTML form
 */
export const LoginForm: FC<{ redirectTo?: string }> = ({ redirectTo }) => {
  // useAuth hook provides the current state AND the login method to update it
  const { login, authenticatedUser } = useAuth();
  const { mail_available } = useGetServer(null);
  // navigate is a method to change current page location to perform client-side redirections
  const navigate = useNavigate();
  const { notify } = useNotifications();
  // form handler provided by react-hook-form library
  const { handleSubmit, register } = useForm<LoginParams>({
    // use the current authenticatedUser as default username
    defaultValues: { username: authenticatedUser?.username },
  });
  const [reset, setReset] = useState<boolean>(false);
  const [mail, setMail] = useState<string>('');

  // useSendResetMail hook provides the sendResetMail method to send an email
  const { sendResetMail } = useSendResetMail();

  // submit handler function automatically get the form data as param
  const onSubmit: SubmitHandler<LoginParams> = async (data) => {
    // use the provided login method to make sure to update the central authenticatedUser state
    await login(data);
    // TODO handle failed login

    // if a redirectTo prop exists, redirect the user to it
    if (redirectTo) {
      navigate(redirectTo);
    }
  };

  const resetPassword = async () => {
    await sendResetMail(mail);
    setReset(false);
    notify({
      type: 'success',
      message: 'An email has been sent to you with instructions if you have an account',
    });
  };

  return (
    <>
      {!reset && (
        <form onSubmit={handleSubmit(onSubmit)}>
          <input
            className="form-control form-appearance mt-2 w-50"
            type="text"
            {...register('username')}
            placeholder="Username"
          />
          <input
            className="form-control mt-2  w-50"
            type="password"
            {...register('password')}
            placeholder="Password"
          />
          <div className="d-flex justify-content-between w-50">
            <button className="btn btn-primary btn-validation">Login</button>
          </div>
        </form>
      )}
      {mail_available && (
        <a href="#" onClick={() => setReset(!reset)}>
          <span className="ms-1">Reset password</span>
        </a>
      )}
      {reset && (
        <div className="d-flex justify-content-center align-items-center">
          <div className="d-flex flex-row align-items-center">
            <input
              className="form-control form-appearance me-2"
              placeholder="Enter your email"
              onChange={(e) => setMail(e.target.value)}
              value={mail}
              type="email"
            />
            <button className="btn btn-danger" onClick={resetPassword}>
              Reset
            </button>
          </div>
        </div>
      )}
    </>
  );
};
