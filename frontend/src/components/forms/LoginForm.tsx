import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';

import { LoginParams } from '../../core/api';
import { useAuth } from '../../core/auth';

export const LoginForm: FC<{ redirectTo?: string }> = ({ redirectTo }) => {
  const { login, authenticatedUser } = useAuth();
  const navigate = useNavigate();
  const { handleSubmit, register } = useForm<LoginParams>({
    defaultValues: { username: authenticatedUser?.username },
  });

  const onSubmit: SubmitHandler<LoginParams> = async (data) => {
    await login(data);
    if (redirectTo) {
      navigate(redirectTo);
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
