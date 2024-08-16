import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';

import { useAuth } from '../../core/auth';
import { LoginParams } from '../../types';

/**
 * LoginForm
 * @param redirectTo a location where the user must be redirected after successful login
 * @returns a HTML form
 */
export const LoginForm: FC<{ redirectTo?: string }> = ({ redirectTo }) => {
  // useAuth hook provides the current state AND the login method to update it
  const { login, authenticatedUser } = useAuth();
  // navigate is a method to change current page location to perform client-side redirections
  const navigate = useNavigate();

  // form handler provided by react-hook-form library
  const { handleSubmit, register } = useForm<LoginParams>({
    // use the current authenticatedUser as default username
    defaultValues: { username: authenticatedUser?.username },
  });

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

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        className="form-control form-appearance mt-2"
        type="text"
        {...register('username')}
        placeholder="Username"
      />
      <input
        className="form-control mt-2"
        type="password"
        {...register('password')}
        placeholder="Password"
      />
      <button className="btn btn-primary btn-validation">Login</button>
    </form>
  );
};
