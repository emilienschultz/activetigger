import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';

import { LoginParams, login, me } from '../../core/api';
import { useAppContext } from '../../core/context';

export const LoginForm: FC = () => {
  const { setAppContext } = useAppContext();
  const { handleSubmit, register } = useForm<LoginParams>();

  const onSubmit: SubmitHandler<LoginParams> = async (data) => {
    const response = await login(data);
    if (response.access_token) {
      const user = await me(response.access_token);
      if (user) setAppContext({ user: { ...user, access_token: response.access_token } });
      else setAppContext({ user: undefined });
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>Connect to the service</div>
      <input className="form-control form-appearance mt-2" type="text" {...register('username')} />
      <input className="form-control mt-2" type="password" {...register('password')} />
      <button className="btn btn-primary btn-validation">Login</button>
    </form>

    // TODO : rediriger vers l'application si validé ?
  );
};
